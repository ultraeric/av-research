import importlib
import tkinter
import imageio
import torch
import cv2
from PIL import Image, ImageTk
from objects import *
from utils.video import *
from utils.config import Config
from utils.formatting import *
from tasks.lane_detection import LaneDetector
from tasks.e2e import *

config = Config()

Net = importlib.import_module(config['model']['pyPath']).Net

steer_avg, steer_std = None, None
motor_avg, motor_std = None, None
accel_avg, accel_std = None, None
torch.cuda.set_device(config['hardware']['gpu'])

display_dims = (1280, 720)
resolution = (1280, 720)
fps_downsample = 6
downsample_factor = 1./4.
downsampled_resolution = (round(resolution[0] * downsample_factor), round(resolution[1] * downsample_factor))
input_frames = config['constants']['inputFrames']

demo_lenth_factor = 1.


class Demo:
    """
    Demo application, displays video output.
    """
    def __init__(self, beam_path):
        """
        :param beam_path: <str> Path to beam to demo
        """
        self.beam = E2EBeam(filepath=beam_path, demo=True)
        self.bricks = self.beam.get_bricks(sort_by=lambda brick: brick.frame)
        self.vid_path = self.beam.source_path

        self.vid_reader = imageio.get_reader(self.vid_path, 'ffmpeg')
        self.vid_writer = imageio.get_writer('./demo_normal.mp4', fps=self.vid_reader.get_meta_data()['fps'])
        self.vid_frames = get_vid_frames(self.vid_reader,
                                         num_frames=round(self.beam.num_frames * fps_downsample),
                                         resolution=resolution,
                                         downsample_factor=1.,
                                         fps_downsample=1)

        self.downsampled_vid_frames = get_vid_frames(num_frames=round(self.beam.num_frames * fps_downsample * demo_lenth_factor),
                                                     vid_frames=self.vid_frames,
                                                     resolution=resolution,
                                                     downsample_factor=downsample_factor,
                                                     fps_downsample=1)

        self.fps = self.vid_reader.get_meta_data()['fps']
        self.controls = []

        self.n_frames_feed = config['constants']['inputFrames']

        self.lane_detector = LaneDetector()

        # Load network
        self.net = Net(config=config).cuda()
        self.net.load(config['model']['saveDirPath'], config['model']['name'], 'core', config['demo']['epoch'])
        self.task = E2ETask(name='e2e', config=config, load_dataset=False)
        self.net.eval()
        self.task.eval()
        self.task.net.load(config['model']['saveDirPath'], config['model']['name'], self.task.name, config['demo']['epoch'])
        self.task.cuda()

        self.calc_controls()
        self.curr_img, self.imgtk = None, None
        self.curr_frame_i = 0

        # Set up tkinter
        self.root = tkinter.Tk()
        self.root.title('Demo')
        self.root.geometry(str(display_dims[0]) + 'x' + str(display_dims[1]))
        self.canvas = tkinter.Canvas(self.root, width=display_dims[0], height=display_dims[1])
        self.canvas.place(x=0, y=0)
        self.canvas_img = None
        self.overlay_line = self.canvas.create_line(display_dims[0] // 2,
                                                    display_dims[1] // 2,
                                                    display_dims[0] // 2,
                                                    display_dims[1] // 2,
                                                    fill='red',
                                                    arrow='last')
        self.root.config(cursor="arrow")

        self.display_loop()

    def get_video(self, frame_i):
        if frame_i < (self.n_frames_feed - 1) * fps_downsample:
            return None
        else:
            vid_frames = get_vid_frames(num_frames=input_frames,
                                        vid_frames=self.downsampled_vid_frames,
                                        resolution=downsampled_resolution,
                                        downsample_factor=1.,
                                        fps_downsample=fps_downsample,
                                        start_frame=frame_i - ((self.n_frames_feed - 1) * fps_downsample))
            return vid_frames

    def get_metadata(self, frame_i):
        for i in range(len(self.bricks) - 1):
            curr_brick, next_brick = self.bricks[i], self.bricks[i + 1]
            if curr_brick.frame * fps_downsample <= frame_i < next_brick.frame * fps_downsample:
                return torch.LongTensor(np.array([0]))
        return torch.LongTensor(np.array([0]))

    def calc_controls(self):
        for frame_i in range(round(36 * self.fps * demo_lenth_factor)):
            vid_frames = self.get_video(frame_i)
            if frame_i % 100 == 0:
                print(frame_i)
            if vid_frames is None:
                self.controls.append((0, 0, 0))
            else:
                metadata = torch.unsqueeze(self.get_metadata(frame_i).cuda(), 0)
                vid_frames = torch.unsqueeze(torch.FloatTensor(vid_frames).cuda(), 0)
                embedding = self.net(vid_frames, metadata)
                output = self.task.net(embedding).data[0]
                controls = (output[0], output[1], output[2])
                self.controls.append(controls)

    def display_loop(self):
        curr_control = self.get_control(self.curr_frame_i)
        self.curr_img = self.vid_frames[self.curr_frame_i]
        self.curr_img = Image.fromarray(
            self.paint_overlay_line(
                self.lane_detector.detect(
                    self.curr_img, roi_only=True),
                curr_control[0], curr_control[1]))
        imgtk = ImageTk.PhotoImage(image=self.curr_img)
        self.imgtk = imgtk
        if self.canvas_img is not None:
            self.canvas.delete(self.canvas_img)
        self.canvas_img = self.canvas.create_image(display_dims[0] // 2, display_dims[1] // 2, image=self.imgtk)
        # self.calc_overlay_line(curr_control[0], curr_control[1])
        self.root.after(17, self.display_loop)
        self.curr_frame_i = (self.curr_frame_i + 1) % round(self.beam.num_frames * fps_downsample * demo_lenth_factor)
        self.vid_writer.append_data(np.array(np.array(self.curr_img)))

    def calc_overlay_line(self, steer, accel):
        x_0, y_0 = display_dims[0] // 2, display_dims[1] // 2
        x_1, y_1 = x_0 + steer * 25, y_0 - round((accel * 1.14) * 200)
        self.canvas.delete(self.overlay_line)
        self.overlay_line = self.canvas.create_line(x_0, y_0, x_1, y_1, fill='red', arrow='last')

    def paint_overlay_line(self, image, steer, accel):
        print(steer, accel)
        x_0, y_0 = image.shape[1] // 2, image.shape[0] // 2
        x_1, y_1 = round(x_0 + steer * 50), y_0 - round((accel * 1.14) * 200)
        cv2.arrowedLine(image, (x_0, y_0), (x_1, y_1), (255, 0, 0), 3)
        return image

    def get_control(self, frame_i):
        control = [0, 0, 0]
        total_weight = 0
        for i in range(14, -1, -1):
            curr_i = frame_i - i
            curr_control = self.controls[curr_i]
            curr_weight = 1. / (i + 1.) ** 0.5
            control[0] += curr_weight * curr_control[0]
            control[1] += curr_weight * curr_control[1]
            control[2] += curr_weight * curr_control[2]
            total_weight += curr_weight
        control = [_/total_weight for _ in control]
        return control

app = Demo('/data/datasets/processed/beams/294.beam')
app.root.mainloop()
