import os
from utils.config import Config
import numpy as np


def _format_model_path(epoch: int, save_dir: str, model_name: str, extensions: str='') -> str:
    """
    Formats the model path.

    :param save_dir: <str> directory of save
    :param model_name: <str> model name
    :param extensions: <str> any extension to the name to add
    :param epoch: <int> epoch
    :return: str
    """
    return os.path.join(save_dir, model_name + extensions + '_epoch_%0.5d.weights' % epoch)


def format_model_path(epoch: int, save_dir: str='', model_name: str='', extensions: str='', config: Config=None) -> str:
    """
    Formats model path with either direct parameters or a model config

    :param save_dir: directory of save
    :param model_name: <str> model name
    :param extensions: <str> any extension to the name to add
    :param epoch: <int> epoch
    :param config: <Config> a config object
    :return: str
    """
    assert bool(config) ^ bool(save_dir and model_name), \
        'Either a <Config> object or model path parameters <save_dir> and <model_name> must be provided'
    assert epoch >= 0, 'Please provide the epoch training the model is on'

    if config:
        return _format_model_path(epoch, config['model']['saveDirPath'], config['model']['name'], extensions)
    else:
        return _format_model_path(epoch, save_dir, model_name, extensions)


def format_loss_path(train: bool, save_dir: str='', model_name: str='', config: Config=None) -> str:
    """
    Formats path of loss file.

    :param train:
    :param save_dir:
    :param model_name:
    :param extensions:
    :param config:
    :return:
    """
    assert bool(config) ^ bool(save_dir and model_name), \
        'Either a <Config> object or model path parameters <save_dir> and <model_name> must be provided'
    suffix = '_train' if train else '_val'
    model_name = config['model']['name'] + suffix + '.loss'
    return os.path.join(config['logging']['dirPath'] if config else save_dir, model_name)


def cuda_to_list(arr):
    return arr.cpu().float().data.numpy().tolist()