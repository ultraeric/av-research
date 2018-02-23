import scipy.ndimage
import numpy as np
import random

def get_vid_frames(imageio_reader=None, num_frames=None, vid_frames=None, resolution=(1280, 720),
                   downsample_factor = 1., fps_downsample=6, start_frame=0):
    """
    Gets video frames from a reader given some information

    :param imageio_reader: imageio reader object
    :param num_frames: number of frames to extract
    :param resolution: (x, y) resolution of source video
    :param downsample_factor: factor to downsample the image
    :param fps_downsample: stride to sample frames from
    :return: frames
    """

    x_len, y_len = resolution[0] * downsample_factor, resolution[1] * downsample_factor
    x_len, y_len = int(x_len), int(y_len)

    # Empty array to store video data in
    processed_vid_frames = np.empty([num_frames, y_len, x_len, 3]).astype(np.uint8)


    for frame_i in range(num_frames):
        if vid_frames is not None:
            frame_data = np.array(vid_frames[start_frame + frame_i * fps_downsample])
        else:
            frame_data = imageio_reader.get_data(start_frame + frame_i * fps_downsample)
        if downsample_factor != 1:
            frame_data = scipy.ndimage.zoom(frame_data, [downsample_factor, downsample_factor, 1.], order=1)
        processed_vid_frames[frame_i] = frame_data

    return processed_vid_frames


def get_steering(course_0, course_1, time_0, time_1):
    """
    Gets steering given direction information and time information

    :param course_0: initial direction
    :param course_1: final direction
    :param time_0: initial time in seconds
    :param time_1: final time in seconds
    :return: steering delta (angular velocity)
    """
    course_0, course_1 = (course_0 + 3600000) % 360, (course_1 + 3600000) % 360

    if course_1 - course_0 > 180.:
        course_0 += 360.
    elif course_0 - course_1 > 180.:
        course_1 += 360.

    if float(time_1 - time_0) == 0:
        return 0.

    angular_v = float(course_1 - course_0) / float(time_1 - time_0)
    return angular_v


def get_accel(speed_0, speed_1, time_0, time_1):
    if float(time_1 - time_0) == 0:
        return 0.
    return float(speed_1 - speed_0) / float(time_1 - time_0)


def detect_steering_angle(beam):
    # Return if steering alread detected
    if len(beam['bricks']) == 0 or 'steering' in beam['bricks'][0].keys():
        beam.postprocess_bricks()
        return beam
    bricks = beam.get_all_bricks(include_vestigial=True, sorted=True)
    for i2 in range(len(bricks) - 1):
        curr_brick = bricks[i2]
        next_brick = bricks[i2 + 1]
        steering = get_steering(curr_brick['rawMetadata']['course'],
                                next_brick['rawMetadata']['course'],
                                curr_brick['rawMetadata']['timestamp'] / 1000.,
                                next_brick['rawMetadata']['timestamp'] / 1000.)
        curr_brick['steering'] = steering
    beam.postprocess_bricks()
    return beam


def detect_accel(beam):
    # Return if steering alread detected
    if len(beam['bricks']) == 0 or 'accel' in beam['bricks'][0].keys():
        beam.postprocess_bricks()
        return beam
    bricks = beam.get_all_bricks(include_vestigial=True, sorted=True)
    for i2 in range(len(bricks) - 1):
        curr_brick = bricks[i2]
        next_brick = bricks[i2 + 1]
        accel = get_accel(curr_brick['rawMetadata']['speed'],
                                next_brick['rawMetadata']['speed'],
                                curr_brick['rawMetadata']['timestamp'] / 1000.,
                                next_brick['rawMetadata']['timestamp'] / 1000.)
        curr_brick['accel'] = accel
    beam.postprocess_bricks()
    return beam


def detect_turn(beam):
    """
    Detects a turn by checking steering. Smears action over timesteps.

    :param beam: beam to process
    :return:
    """
    if len(beam['bricks']) == 0:
        return beam
    bricks = beam['bricks']
    for brick in bricks:
        action = 'None'
        if 'steering' not in brick.keys():
            break
        elif brick['steering'] >= 15:
            action = 'rightTurn'
        elif brick['steering'] <= -15:
            action = 'leftTurn'
        for compare_brick in bricks:
            time_diff = (brick['timestamp'] - compare_brick['timestamp']) / 1000.
            if -2.1 <= time_diff <= 3.1:
                compare_brick['actions'].append(action) if action not in compare_brick['actions'] else None
    return beam


def match_to_frame(fps, timestamp, startstamp):
    """
    Matches a given timestamp to a frame, zero-indexed.

    :param fps: fps
    :param timestamp: timestamp of metadata in millis
    :param startstamp: starting timestamp in millis
    :return: int
    """
    return round((timestamp - startstamp) * fps / 1000.)
