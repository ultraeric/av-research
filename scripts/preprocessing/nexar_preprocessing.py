import imageio
import os
import json
import queue
import traceback
from utils import *
from threading import Lock
from threading import Thread
from objects import *

fps_downsample = 6
max_time = 36
downsample_factor = 1. / 4.
num_threads = 8
files_per_thread = 8
resolution = (1280, 720)

def preprocess(nexar_root='..', dataset_root='..'):
    # Set up directories
    gps_dir = nexar_root + '/GPS+IMU/Files/'
    vid_dir = nexar_root + '/HDVideos/'
    beam_dir = dataset_root + '/beams/'
    hdf5_dir = dataset_root + '/hdf5/'

    counterlock = Lock()
    i = 0

    # Create directories
    if not os.path.exists(beam_dir):
        os.makedirs(beam_dir)
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)

    def get_next_file_index():
        counterlock.acquire()
        nonlocal i
        local_i = i
        i += 1
        counterlock.release()
        return local_i

    class Worker(Thread):
        def __init__(self, thread_num=0, gps_q=queue.Queue()):
            super().__init__()
            self.file_queue = gps_q
            self.dataset_filepaths = [hdf5_dir + str(thread_num) + '_' + str(i) + '.hdf5'
                                      for i in range(files_per_thread)]

        def run(self):
            self_i = 0
            while not self.file_queue.empty():
                try:
                    gps_name = self.file_queue.get()
                    local_i = get_next_file_index()
                    if local_i % 5 == 0:
                        print('Processing video {}'.format(local_i))

                    vid_name = gps_name[33:-5] + '.mov'

                    # Skip videos that are too short or have improper ratios
                    vid_reader = imageio.get_reader(vid_dir + vid_name, 'ffmpeg')
                    if vid_reader.get_meta_data()['duration'] < 38:
                        continue
                    if not (vid_reader.get_meta_data()['source_size'] == resolution
                            or vid_reader.get_meta_data()['source_size'] == (resolution[1], resolution[0])):
                        continue

                    # Get the metadata
                    with open(gps_dir + gps_name) as f:
                        metadata = json.load(f)
                    fps = vid_reader.get_meta_data()['fps'] / fps_downsample
                    start_time = metadata[0]['timestamp']

                    # Initialize beam object
                    beam_path = beam_dir + str(local_i) + '.beam'
                    curr_beam = Beam(data=
                                    {'name': str(local_i),
                                      'id': gps_name[:-5],
                                      'datasetId': gps_name[:-5],
                                      'filepath': beam_path,
                                      'sourcePath': vid_dir + vid_name,
                                      'sourceJsonPath': gps_dir + gps_name,
                                      'timestamp': start_time,
                                      'resolution': vid_reader.get_meta_data()['source_size'],
                                      'duration': max_time,
                                      'fps': fps,
                                      'numFrames': max_time * fps})

                    nframes = round(fps * max_time)

                    # Extract video data
                    vid_frames = get_vid_frames(vid_reader,
                                                nframes,
                                                resolution=resolution,
                                                downsample_factor=downsample_factor,
                                                fps_downsample=fps_downsample)

                    # Add all bricks in this file
                    for datapoint in metadata:
                        frame_i = match_to_frame(fps, datapoint['timestamp'], start_time)
                        if frame_i >= nframes:
                            break
                        brick = Brick({'frame': frame_i,
                                       'motor': datapoint['speed'],
                                       'actions': [],
                                       'timestamp': datapoint['timestamp'],
                                       'rawMetadata': datapoint})
                        curr_beam.add_brick(brick)

                    # Reprocess bricks
                    detect_steering_angle(curr_beam)
                    detect_turn(curr_beam)
                    curr_beam.postprocess_bricks()

                    curr_beam['hdf5Path'] = self.dataset_filepaths[self_i]
                    curr_beam['numBricks'] = len(curr_beam['bricks'])

                    # Serialize beams and video
                    curr_beam.serialize(beam_path)
                    curr_beam.write_data(vid_frames)
                    self_i = (self_i + 1) % files_per_thread
                except Exception:
                    print(traceback.format_exc())

    gps_list = list(os.listdir(gps_dir))

    # Remove anything already seen
    beam_filenames = list(os.listdir(beam_dir))
    remove_gps_list = []
    max_i = 0
    for _ in range(len(beam_filenames)):
        beam_filename = beam_filenames[_]
        max_i = max(max_i, int(beam_filename[:-5]))
        beam = Beam(filepath=beam_dir + '/' + beam_filename)
        if _ % 500 == 0:
            print(_)
        for gps_name in gps_list:
            vid_name = gps_name[33:-5] + '.mov'
            if beam['sourcePath'] == vid_name:
                remove_gps_list.append(gps_name)
    gps_list = [item for item in gps_list if item not in remove_gps_list]

    i = max_i + 1

    gps_queue = queue.Queue()
    for gps_name in gps_list:
        gps_queue.put(gps_name)

    print(i, len(gps_list))

    for thread in range(num_threads):
        worker = Worker(thread, gps_queue)
        worker.start()
