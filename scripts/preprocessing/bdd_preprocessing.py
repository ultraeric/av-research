import os
import queue
import traceback
import numpy as np
from utils import *
from threading import Lock
from threading import Thread
from objects import *
from h5py import File
from os import listdir, walk
from hashlib import sha256

num_threads = 12
files_per_thread = 8
video_file_name = 'images.h5py'
metadata_file_name = 'metadata.h5py'

class Run:
    def __init__(self, dir='', dest_dir=''):
        self.dir = dir
        self.dest_dir = dest_dir
        dirfiles = next(walk(self.dir))[1]
        self.files = []

        self.run_labels = []
        run_labels_file = File(os.path.join(dir, 'run_labels.h5py'))
        for key in list(run_labels_file):
            if run_labels_file[key][0]:
                self.run_labels.append(key)

        for dir in dirfiles:
            sub_run_dir = os.path.join(self.dir, dir)
            video_file = File(os.path.join(sub_run_dir, video_file_name), mode='r')
            metadata_file = File(os.path.join(sub_run_dir, metadata_file_name), mode='r')
            self.files.append((sub_run_dir, video_file, metadata_file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.files[item]

    def read_video(self, index):
        left_video = self.files[index][1]['left']
        right_video = self.files[index][1]['right']
        video = np.concatenate([left_video, right_video], axis=3)
        return video

    def read_metadata(self, index, frame):
        #TODO: Return (steering, motor, mode) tuple
        metadata_file = self.files[index][2]
        return (metadata_file['steer'][frame], metadata_file['motor'][frame], metadata_file['state'][frame])

    def get_brick(self, index, frame):
        steering, motor, mode = self.read_metadata(index, frame)
        brick = Brick(json_dict={'frame': frame,
                                 'truth': [steering, motor],
                                 'metadata': {'mode': mode},
                                 'rawMetadata': {}})
        return brick

    def get_beam(self, index, local_i, hdf5_path):
        curr_file = self.files[index]
        beam_name = sha256((curr_file[0] + str(local_i)).encode('utf-8')).hexdigest()
        beam_id = sha256(beam_name.encode('utf-8')).hexdigest()
        nframes = curr_file[1]['left'].shape[0]
        beam_dict = {'name': beam_name,
                     'id': beam_id,
                     'datasetId': beam_id,
                     'filepath': os.path.join(self.dest_dir, str(local_i) + '.beam'),
                     'numFrames': nframes,
                     'fps': 30,
                     }
        curr_beam = Beam(Brick, json_dict=beam_dict)
        video = self.read_video(index)
        for frame_i in range(nframes):
            metadata = self.read_metadata(index, frame_i)
            brick = Brick({'frame': frame_i,
                           'truth': {'steering': int(metadata[0]),
                                     'motor': int(metadata[1]),
                                     'mode': int(metadata[2])},
                           'metadata': {},
                           'rawMetadata': {}})
            curr_beam.add_brick(brick)
        curr_beam.hdf5_path = hdf5_path
        curr_beam.write_data(video)
        return curr_beam



def preprocess(raw_root='/data/dataset/bair_car_data_Main_Dataset/processed_h5py',
               dataset_root='/data/dataset/bair_car_data_Main_Dataset/public_dataset'):
    # Set up directories
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
        def __init__(self, thread_num=0, run_q=queue.Queue()):
            super().__init__()
            self.run_queue = run_q
            self.hdf5_filepaths = [hdf5_dir + str(thread_num) + '_' + str(i) + '.hdf5'
                                      for i in range(files_per_thread)]

        def run(self):
            self_i = 0
            while not self.run_queue.empty():
                try:
                    run = self.run_queue.get()

                    run = Run(run, os.path.join(dataset_root, 'beams'))
                    hdf5_path = self.hdf5_filepaths[self_i]
                    for i in range(len(run)):
                        local_i = get_next_file_index()
                        if local_i % 5 == 0:
                            print('Processing video {}'.format(local_i))
                        curr_beam = run.get_beam(i, local_i, hdf5_path)
                        curr_beam.serialize(curr_beam.filepath)

                    # Serialize beams and video
                    self_i = (self_i + 1) % files_per_thread
                except Exception:
                    print(traceback.format_exc())


    run_queue = queue.Queue()
    for run_name in next(walk(raw_root))[1]:
        run_queue.put(os.path.join(raw_root, run_name))

    for thread in range(num_threads):
        worker = Worker(thread, run_queue)
        worker.start()
