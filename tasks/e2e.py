import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Iterable
from objects import *
from objects._session import session
from utils.config import Config
from nets.tasks.e2e import E2ENet


class E2EBeam(Beam):
    def __init__(self, *args, **kwargs):
        self.timestamp = ''
        self.duration, self.fps, self.num_bricks, self.num_frames = 0, 0, 0, 0
        self.resolution = [0, 0]
        super().__init__(E2EBrick, *args, **kwargs)

    @property
    def valid(self) -> bool:
        return abs(self.beam.config['constants']['runningFps'] - self.fps) / self.beam.config['constants']['runningFps'] < 0.1


class E2EBrick(Brick):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def valid(self) -> bool:
        steering, accel = self.truth['steering'], self.truth['accel']
        if steering is None or accel is None:
            validity = False
        else:
            validity = -90. <= steering <= 90. and -20. <= accel <= 20.
        return validity

    def get_input(self):
        input_data = session.read_hdf5(filepath=self.beam.hdf5_path,
                                       dataset_id=self.beam.dataset_id,
                                       start_frame=self.frame - self.beam.config['constants']['runningFps'] + 1,
                                       end_frame=self.frame + 1)
        return torch.FloatTensor(input_data)

    def get_metadata(self):
        rand_state = random.getstate()
        random.seed()

        action = random.choice(self.metadata['actions'])
        steering_meta = -1

        if 'rightTurn' == action:
            steering_meta = 1
        elif 'leftTurn' == action:
            steering_meta = 2
        elif 'None' == action:
            steering_meta = 0

        random.setstate(rand_state)
        return torch.LongTensor(np.array([steering_meta]))

    def get_truth(self):
        steering, accel, motor = self.truth['steering'], self.truth['accel'], self.truth['motor']
        return torch.FloatTensor(np.array([steering, accel, motor]))


class E2EDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stds = torch.FloatTensor(self._normalize_bricks(self.train_bricks))

    def _normalize_bricks(self, train_bricks=None):
        train_bricks = train_bricks or self.train_bricks
        steering = np.array([brick.truth['steering'] for brick in train_bricks])
        accel = np.array([brick.truth['accel'] for brick in train_bricks])
        motor = np.array([brick.truth['motor'] for brick in train_bricks])
        return np.array([np.std(steering), np.std(accel), np.std(motor)])

    def get_posthook(self, input, metadata, truth):
        return input, metadata, truth / self._stds

    def load_beam(self, filepath: str) -> Iterable[Brick]:
        beam = E2EBeam(filepath=filepath, config=self.config)
        bricks = beam.get_bricks(exclude_invalid=True,
                                 sort_by=lambda brick: brick.frame,
                                 filter_by=lambda brick: (brick.frame >= beam.config['constants']['inputFrames']),
                                 filter_posthook=lambda bricks: bricks[:-1])
        return bricks


class E2ETask(Task):
    def __init__(self,  name: str, config: Config, load_dataset=True):
        dataset = E2EDataset(config=config, dirpath=config['tasks']['e2e']['dataset']['dirPath']) if load_dataset else None
        opt_kwargs = {'eps': 1e-4}
        super().__init__(name, config, dataset, E2ENet, nn.MSELoss, optim.Adam, opt_kwargs=opt_kwargs)


Task = E2ETask
