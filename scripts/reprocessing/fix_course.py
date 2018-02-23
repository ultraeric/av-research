import os
from objects import *
from utils import *

def reprocess(dataset_root='..'):
    beam_dir = dataset_root + '/beams/'
    beam_filenames = list(os.listdir(beam_dir))
    for i in range(len(beam_filenames)):
        beam_filename = beam_filenames[i]
        beam = Beam(filepath=beam_dir + '/' + beam_filename)
        bricks = beam['bricks']
        new_bricks = []
        for i2 in range(len(bricks) - 1):
            curr_brick = bricks[i2]
            next_brick = bricks[i2 + 1]
            meta_key = 'raw_metadata' if 'raw_metadata' in curr_brick.keys() else 'rawMetadata'
            steering = get_steering(curr_brick['steering'], next_brick['steering'],
                                    curr_brick[meta_key]['timestamp'] / 1000.,
                                    next_brick[meta_key]['timestamp'] / 1000.)
            curr_brick['steering'] = steering
            curr_brick['rawMetadata'] = curr_brick[meta_key]
            if 'raw_metadata' in curr_brick.keys():
                del curr_brick['raw_metadata']
            new_bricks.append(curr_brick)
        beam.replace_bricks(new_bricks=new_bricks)
        beam.serialize()
        if i % 250 == 0:
            print(i)
