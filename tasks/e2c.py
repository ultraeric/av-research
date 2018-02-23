import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Iterable
from tasks.e2e import E2EBrick
from nets.tasks.e2c import E2CNet
from objects.dataset import Dataset
from objects.brick import Brick
from objects.beam import Beam
from objects.task import Task
from utils.config import Config


class E2CBeam(Beam):
    def __init__(self, *args, **kwargs):
        self.timestamp = ''
        self.duration, self.fps, self.num_bricks, self.num_frames = 0, 0, 0, 0
        self.resolution = [0, 0]
        super().__init__(E2CBrick, *args, **kwargs)

    @property
    def valid(self) -> bool:
        return abs(self.beam.config['constants']['runningFps'] - self.fps) / self.beam.config['constants']['runningFps'] < 0.1

class E2CBrick(E2EBrick):
    def get_truth(self):
        accel = self.truth['accel']
        rand_state = random.getstate()
        random.seed()

        action = random.choice(self.metadata['actions'])
        action_class = -1

        if 'rightTurn' == action:
            action_class = 1
        elif 'leftTurn' == action:
            action_class = 2
        elif 'None' == action:
            action_class = 0
            if accel >= 2.5:
                action_class = 3
            elif accel <= -2.5:
                action_class = 4
        random.setstate(rand_state)
        return action_class


class E2CDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_weights = torch.FloatTensor(self._get_action_weights(self.train_bricks))

    def _get_action_weights(self, train_bricks=None):
        train_bricks = train_bricks or self.train_bricks
        counts = [0, 0, 0, 0, 0]
        for brick in train_bricks:
            counts[brick.get_truth()] += 1
        inv_freq = sum(counts) / np.array(counts, dtype=np.float)
        weights = len(counts) * inv_freq / sum(inv_freq)
        print(weights)
        return weights

    def load_beam(self, filepath: str) -> Iterable[Brick]:
        beam = E2CBeam(filepath=filepath, config=self.config)
        bricks = beam.get_bricks(exclude_invalid=True,
                                 sort_by=lambda brick: brick.frame,
                                 filter_by=lambda brick: (brick.frame >= beam.config['constants']['inputFrames']),
                                 filter_posthook=lambda bricks: bricks[:-1])
        return bricks


class E2CTask(Task):
    def __init__(self,  name: str, config: Config, load_dataset=True):
        dataset = E2CDataset(config=config, dirpath=config['tasks']['e2c']['dataset']['dirPath']) if load_dataset else None
        loss_kwargs = {'weight': dataset.action_weights}
        opt_kwargs = {'eps': 1e-4}
        super().__init__(name, config, dataset, E2CNet, nn.CrossEntropyLoss, optim.Adam, loss_kwargs, opt_kwargs)


Task = E2CTask
