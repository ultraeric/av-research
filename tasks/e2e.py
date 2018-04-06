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
        super().__init__(E2EBrick, *args, **kwargs)

    @property
    def valid(self) -> bool:
        return True


class E2EBrick(Brick):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def valid(self) -> bool:
        return self.beam.backward_brick_validity(self.frame,
                                                 fps=self.beam.config['training']['inputFPS'],
                                                 num_frames=self.beam.config['training']['inputFrames']) and \
               self.beam.forward_brick_validity(self.frame,
                                                 fps=self.beam.config['training']['outputFPS'],
                                                 num_frames=self.beam.config['training']['outputFrames'])

    def get_input(self):
        input_data = self.beam.read_data(self.frame,
                                         fps=self.beam.config['training']['inputFPS'],
                                         num_frames=self.beam.config['training']['inputFrames'])
        return torch.FloatTensor(input_data)

    def get_metadata(self):
        return torch.LongTensor(np.array([0]))

    def get_truth(self):
        steering, motor = self.truth['steering'], self.truth['motor']
        truth_arr = [steering, motor]
        forward_bricks = self.beam.get_forward_bricks(self.frame,
                                                         fps=self.beam.config['training']['inputFPS'],
                                                         num_frames=self.beam.config['training']['inputFrames'])
        for brick in forward_bricks:
            truth_arr.extend([brick.truth['steering'], brick.truth['motor']])
        return torch.FloatTensor(np.array(truth_arr))


class E2EDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stds = torch.FloatTensor(self._normalize_bricks(self.train_bricks))

    def _normalize_bricks(self, train_bricks=None):
        train_bricks = train_bricks or self.train_bricks
        steering = np.array([brick.truth['steering'] for brick in train_bricks])
        motor = np.array([brick.truth['motor'] for brick in train_bricks])
        return np.array([np.std(steering), np.std(motor)])

    def get_posthook(self, input, metadata, truth):
        return input, metadata, truth / self._stds

    def load_beam(self, filepath: str) -> Iterable[Brick]:
        beam = E2EBeam(filepath=filepath, config=self.config)
        bricks = beam.get_bricks(exclude_invalid=True,
                                 sort_by=lambda brick: brick.frame,
                                 filter_posthook=lambda bricks: bricks[:-1])
        return bricks


class E2ETask(Task):
    def __init__(self,  name: str, config: Config, load_dataset=True):
        dataset = E2EDataset(config=config, dirpath=config['tasks']['e2e']['dataset']['dirPath']) if load_dataset else None
        opt_kwargs = {'eps': 1e-4}
        super().__init__(name, config, dataset, E2ENet, nn.MSELoss, optim.Adam, opt_kwargs=opt_kwargs)


Task = E2ETask
