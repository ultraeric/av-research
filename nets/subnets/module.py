import torch.nn as nn
import torch
import os.path
import os


class Module(nn.Module):
    """
    Wraps the nn module with some helper methods for all conv-based modules
    """
    def __init__(self, config=None):
        """
        Wrapper init. Doesn't do much.
        """
        super().__init__()

    def format_path(self, dir_path: str, core_net: str, task: str, epoch: int):
        dir_path = os.path.join(dir_path, core_net)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return os.path.join(dir_path, task + '_epoch{0:05d}.nn'.format(epoch))

    def save(self, dir_path: str, core_net: str, task: str, epoch: int):
        filepath = self.format_path(dir_path, core_net, task, epoch)
        torch.save(
            self.state_dict(),
            filepath
        )
        return self

    def load(self, dir_path: str, core_net: str, task: str, epoch: int):
        filepath = self.format_path(dir_path, core_net, task, epoch)
        model_data = torch.load(filepath)
        self.load_state_dict(model_data)
        return self

    def format_frames(self, input, time_dim=-1, feat_dim=-1):
        """
        Assumes that this is given the format [batch, time, height, width, features]

        :param input: tensor to transform
        :param time_dimension: time dimension
        :param conv_dimension: dimension to put convolutions into
        :return: transformed tensor
        """
        default_time_dim, default_height_dim, default_width_dim, default_feat_dim = 1, 2, 3, 4
        input = input.contiguous()

        # If there shouldn't be a time dimension, remove it
        if time_dim == -1:
            inputs = torch.unbind(input, 1)
            default_feat_dim -= 1
            default_height_dim -= 1
            default_width_dim -= 1
            input = torch.cat(inputs, dim=default_feat_dim).contiguous()

        # If there is a requested feature dimension, put it there. Otherwise, put it in dimension default_feat_dim-2.
        if feat_dim != -1:
            input = torch.transpose(input, default_height_dim, default_width_dim).contiguous()
            input = torch.transpose(input, default_height_dim, default_feat_dim).contiguous()

        return input
