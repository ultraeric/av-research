try:
    import ujson as json
except:
    import json
import os
from os.path import expanduser
from objects import *
from utils.video import *
from tasks.e2e import *

dataset_root = '/data/datasets/processed'
beam_dir = dataset_root + '/beams/'
nexar_root = expanduser('ROOT TO BDDNexar')
json_dir = nexar_root + 'GPS+IMU/Files/'
vid_dir = nexar_root + '/HDVideos/'
i = 0


for beam_filename in list(os.listdir(beam_dir)):
    with open(beam_dir + '/' + beam_filename) as f:
        json_dict = json.load(f)
    json_dict['bricks'].append(json_dict['vestigialBrick'])
    del json_dict['vestigialBrick']
    for brick_json in json_dict['bricks']:
        brick_json['truth'] = {'steering': brick_json.get('steering', None),
                               'motor': brick_json.get('motor', None),
                               'accel': brick_json.get('accel', None)}
        brick_json.pop('steering', None)
        brick_json.pop('motor', None)
        brick_json.pop('accel', None)
        brick_json['metadata'] = {'actions': brick_json.get('actions', [])}
        brick_json.pop('actions', None)
    beam = E2EBeam(json_dict=json_dict, config=True)
    # for brick in beam['bricks']:
    #     brick['actions'] = []
    # beam['sourceJsonPath'] = json_dir + beam['id'] + '.json'

    # Get the metadata
    # with open(beam['sourceJsonPath']) as f:
    #     metadata = json.load(f)

    # start_time = metadata[0]['timestamp']
    # fps = beam['fps']
    # nframes = beam['numFrames']
    # beam.clear_bricks()
    #
    # # Add all bricks in this file
    # for datapoint in metadata:
    #     frame_i = match_to_frame(fps, datapoint['timestamp'], start_time)
    #     if frame_i >= round(nframes):
    #         break
    #     brick = Brick({'frame': frame_i,
    #                    'motor': datapoint['speed'],
    #                    'actions': [],
    #                    'timestamp': datapoint['timestamp'],
    #                    'rawMetadata': datapoint})
    #     beam.add_brick(brick)

    # Reprocess bricks
    # detect_steering_angle(beam)
    # detect_turn(beam)
    # detect_accel(beam)
    # beam.postprocess_bricks()

    # assert all([True if 'steering' in brick.keys() else False for brick in beam.get_all_bricks()])

    beam.serialize(filepath=beam.filepath)
    if i % 500 == 0:
        print(i)
    i += 1