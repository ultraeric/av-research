import torch.optim as optim
from abc import abstractmethod
from typing import List

from torch import nn
import random
from objects.dataset import Dataset
from utils.config import Config
from nets.subnets.module import Module


class Task:
    """
    The root class for all tasks. Defines a few required methods.
    """
    def __init__(self,
                 name: str,
                 config: Config,
                 dataset: Dataset,
                 Net: Module,
                 Loss: nn.Module,
                 Optimizer=optim.SGD,
                 loss_kwargs={},
                 opt_kwargs={'weight_decay': 1e-4}):
        self.inst_config = config['training']['tasks'][name]

        self.name = name
        self.config = config
        self.dataset = dataset
        if self.dataset:
            self.dataset._frequency = self.inst_config['frequency']
        self.net = Net(config=config)
        self.loss_func = Loss(**loss_kwargs)

        self.optimizer = Optimizer(self.net.parameters(), lr=config['training']['learningRate'] or 1e-3, **opt_kwargs)

        self.curr_epoch = -1
        self.curr_train_loader = None
        self.curr_val_loader = None
        self.curr_train_batch_len = 0
        self.curr_val_batch_len = 0

    def save_net(self, epoch):
        self.net.save(self.config['model']['saveDirPath'],
                      self.config['model']['name'],
                      self.name,
                      epoch)

    def load_net(self, epoch):
        self.net.load(self.config['model']['saveDirPath'],
                      self.config['model']['name'],
                      self.name,
                      epoch)

    def init_epoch(self, epoch):
        """
        Initializes parameters for a single epoch for a task. Remember to call when starting an epoch.
        Used in training.
        """
        if epoch == self.curr_epoch:
            return
        else:
            self.curr_epoch = epoch
            self.curr_train_loader = self.dataset.get_train_loader(curr_epoch=epoch)
            self.curr_val_loader = self.dataset.get_val_loader()
            self.curr_train_batch_len = len(self.dataset.curr_train_partition) // self.config['training']['batchSize']
            self.curr_val_batch_len = len(self.dataset.curr_val_partition) // self.config['training']['batchSize']

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def cpu(self):
        self.net = self.net.cpu()
        self.loss_func = self.net.cpu()
        return self

    def cuda(self):
        self.net = self.net.cuda()
        self.loss_func = self.loss_func.cuda()
        return self

    def half(self):
        self.net = self.net.half()
        self.loss_func = self.loss_func.half()
        return self

class TaskManager:
    def __init__(self, tasks=[]):
        self.tasks = tasks

        self.curr_epoch = -1
        self.curr_train_batch_len = 0
        self.curr_train_iters = {}

        self.curr_val_batch_len = 0
        self.curr_val_iters = {}

    def _clear(self):
        self.curr_train_batch_len = 0
        self.curr_train_iters = {}

        self.curr_val_batch_len = 0
        self.curr_val_iters = {}

    def init_epoch(self, epoch):
        """
        Initializes parameters for a single epoch for a list of tasks. Remember to call when starting an epoch.
        Used in training.
        """
        if epoch == self.curr_epoch:
            return
        self._clear()
        for task in self.tasks:
            task.init_epoch(epoch)
            self.curr_train_iters[task] = iter(task.curr_train_loader)
            self.curr_val_iters[task] = iter(task.curr_val_loader)
            self.curr_train_batch_len += task.curr_train_batch_len
            self.curr_val_batch_len += task.curr_val_batch_len

    def get_train_loader(self):
        indices = set(range(self.curr_train_batch_len))
        while len(indices) > 0:
            index = random.sample(indices, 1)[0]
            indices.remove(index)
            task = self.map_train_index_to_task(index)
            yield next(self.curr_train_iters[task]) + [task]
        raise StopIteration()

    def get_val_loader(self):
        indices = set(range(self.curr_val_batch_len))
        while len(indices) > 0:
            index = random.sample(indices, 1)[0]
            indices.remove(index)
            task = self.map_val_index_to_task(index)
            yield next(self.curr_val_iters[task]) + [task]
        raise StopIteration()

    def map_train_index_to_task(self, i):
        for task in self.tasks:
            if i < task.curr_train_batch_len:
                return task
            else:
                i -= task.curr_train_batch_len

    def map_val_index_to_task(self, i):
        for task in self.tasks:
            if i < task.curr_val_batch_len:
                return task
            else:
                i -= task.curr_val_batch_len

    def save_nets(self, epoch):
        for task in self.tasks:
            task.save_net(epoch)

    def load_nets(self, epoch):
        for task in self.tasks:
            task.load_net(epoch)
