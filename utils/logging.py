import time
from utils.config import Config
from utils.formatting import *


class LossLogger:
    """
    Utility class to log losses
    """
    def __init__(self, train: bool, filepath: str='', config: Config=None):
        assert bool(config) ^ bool(filepath), 'Please supply exactly one of <filepath> or <config> parameters.'
        self.filepath = filepath or format_loss_path(train, config=config)
        self.loss_file = open(self.filepath, 'a')
        self.stored_losses = []

    def store(self, loss):
        if loss >= 2.5:
            return
        self.stored_losses.append(loss)

    def flush(self):
        with open(self.filepath, 'a') as loss_file:
            loss_file.write(str(sum(self.stored_losses) / len(self.stored_losses)) + '\n')
            loss_file.flush()
            self.stored_losses = []


class ConsoleLogger:
    """
    Utility class to log to console
    """
    def __init__(self):
        self.debounce_time = time.time()

    def _shound_debounce(self, t, debounce_buffer=1.):
        return t - self.debounce_time < debounce_buffer

    def loss(self, train, epoch, curr_i, max_i, loss, debounce_buffer=1.):
        if self._shound_debounce(time.time(), debounce_buffer):
            return
        self.debounce_time = time.time()
        print(('Train' if train else 'Validation') + ': Epoch {} [{}/{} ({:.1f})%] \tLoss: {:.5f}'
              .format(epoch, curr_i, max_i, 100. * curr_i / max_i, loss))

    def progress(self, base_message, i, total, debounce_buffer=1.):
        if self._shound_debounce(time.time(), debounce_buffer):
            return
        self.debounce_time = time.time()
        print(base_message + ' Progress: {}/{} ({})%'.format(i, total, 100. * i / total))
