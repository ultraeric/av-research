import torch
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import random
from typing import Iterable
from abc import abstractmethod
from .beam import *
from utils.logging import *
from utils.video import *

console_logger = ConsoleLogger()


def should_keep(p_keep=1.):
    """Utility function that determines if this sample should be kept"""
    return random.random() <= p_keep


class Dataset(data.Dataset):
    def __init__(self,
                 config: Config,
                 dirpath: str=None,
                 dirpaths: Iterable[str]=None,
                 frequency: float=1):
        """
        Initializes the dataset.

        :param config: <Config> configuration file to use to initialize dataset
        """
        assert bool(dirpath) ^ bool(dirpaths), 'Please specify exactly one of <dirpath> or <dirpaths>'
        if dirpath:
            dirpaths, dirpath = [dirpath], None

        super().__init__()
        self.config = config

        self._frequency = frequency

        self._constants = config['constants']
        self._training = config['training']

        self._curr_train_partition = []
        self._curr_val_partition = []

        self._train_bricks, self._val_bricks = self._load_beams(dirpaths)

    def __len__(self):
        return len(self._train_bricks) + len(self._val_bricks)

    def __getitem__(self, index):
        # Select either train_data indexing or val_data indexing
        active_dataset = self._train_bricks
        if index >= len(self._train_bricks):
            index -= len(self._train_bricks)
            active_dataset = self._val_bricks

        # Retrieve data
        brick = active_dataset[index]

        input = brick.get_input()
        metadata = brick.get_metadata()
        truth = brick.get_truth()

        return self.get_posthook(input, metadata, truth)

    def _load_beams(self, dirpaths: Iterable[str]) -> [Iterable[Brick], Iterable[Brick]]:
        """
        Helper function. Loads all the beams in a directory into bricks and data

        :param dirpaths: <Iterable<str>> iterable of paths to dir
        :return: <Iterable<Brick>>, <Iterable<Brick>>
        """

        random.seed(self._training['seed'])
        beam_filepaths = []
        for dirpath in dirpaths:
            beam_filepaths.extend([os.path.join(dirpath, filename) for filename in os.listdir(dirpath)])
        train_bricks, val_bricks = [], []

        for i, (filepath) in enumerate(beam_filepaths):
            bricks = self.load_beam(filepath=filepath)
            if should_keep(p_keep=self._training['trainRatio']):
                train_bricks.extend(bricks)
            else:
                val_bricks.extend(bricks)
            console_logger.progress('Loading Beams -', i + 1, len(beam_filepaths), debounce_buffer=5./3.)
        console_logger.progress('Loading Beams -', i + 1, len(beam_filepaths), debounce_buffer=0.)

        random.seed()
        return train_bricks, val_bricks

    def _get_data_loader(self, *args, epoch: int=-1, train: bool=True, **kwargs) -> data.DataLoader:
        """Internal helper method to get a DataLoader"""
        assert not train or epoch >= 0, 'Internal Dataset Error: Provide an <epoch> with <train> set to true.'

        random.seed(self._training['seed'] + epoch)
        # Sets indices to either training indices or validation indices
        data_indices = range(len(self._train_bricks)) if train else range(len(self._train_bricks), len(self))
        # Prunes indices based on subsampling ratio
        data_indices = [index for index in data_indices
                        if should_keep(p_keep=self._training['subsampleRatio'] * self._frequency)]

        if train:
            self._curr_train_partition = data_indices
        else:
            self._curr_val_partition = data_indices

        random.seed()
        kwargs['sampler'] = sampler.SubsetRandomSampler(data_indices)
        kwargs['num_workers'] = self._training['numWorkers']
        kwargs['batch_size'] = self._training['batchSize']
        kwargs['drop_last'] = True
        kwargs.pop('epoch', None)
        kwargs.pop('train', None)

        return data.DataLoader(self, *args, **kwargs) if len(data_indices) > 0 else data_indices

    def get_train_loader(self, curr_epoch, *args, **kwargs) -> data.DataLoader:
        return self._get_data_loader(*args, epoch=curr_epoch, train=True, **kwargs)

    def get_val_loader(self, *args, **kwargs) -> data.DataLoader:
        return self._get_data_loader(*args, train=False, **kwargs)

    @property
    def curr_train_partition(self):
        return self._curr_train_partition

    @property
    def curr_val_partition(self):
        return self._curr_val_partition

    @property
    def train_bricks(self):
        return self._train_bricks

    @property
    def val_bricks(self):
        return self._val_bricks

    @abstractmethod
    def load_beam(self, filepath: str) -> Iterable[Brick]:
        """Loads the bricks for a specific beam and returns it."""

    def get_posthook(self, input, metadata, truth):
        return input, metadata, truth
