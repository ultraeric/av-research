import torch.nn as nn
import torch
import torch.nn.init as init
import math


class Fire(nn.Module):
    """Implementation of Fire module"""

    def __init__(self, in_channels, out_channels, activation=nn.ReLU,
                 squeeze_ratio=0.5, dropout=0., groups=1, batchnorm=True, highway=True):
        """
        Sets up parameters of the fire module

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param activation: activation module to use
        :param squeeze_ratio: ratio of internal squeeze dimensions / out_channels
        :param batchnorm: whether or not to use batchnorm
        :param highway: whether or not to use highway connections
        """
        super().__init__()

        squeeze_channels = int(in_channels * squeeze_ratio)
        dim_1x1 = int(math.ceil(out_channels / 2))
        dim_3x3 = int(math.floor(out_channels / 2))

        self.highway = highway and in_channels == out_channels
        self.batchnorm = batchnorm
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            activation(inplace=True)
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, dim_1x1, kernel_size=1, groups=groups),
            activation(inplace=True)
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, dim_3x3, kernel_size=3, padding=1, groups=groups),
            activation(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0. else None

    def forward(self, input_data):
        """Forward-propagates data through Fire module"""
        output_data = self.squeeze(input_data)
        output_data = torch.cat([
            self.expand1x1(output_data),
            self.expand3x3(output_data)
        ], 1)
        output_data = output_data + input_data if self.highway else output_data
        output_data = self.norm(output_data) if self.batchnorm else output_data
        output_data = self.dropout(output_data) if self.dropout else output_data
        return output_data

class SqueezeSubmodule(nn.Module):
    """Submodule for Squeezenet-based networks"""
    def __init__(self, fire_func=Fire, pool_func=nn.AvgPool2d):
        """

        :param fire_func: <function/class> function for constructing a fire module, if not default
        :param pool_func: <function/class> function to use for pooling
        """
        super().__init__()
        self.conv_net = nn.Sequential(
            fire_func(32, 32, groups=2),
            fire_func(32, 48, groups=2),
            fire_func(48, 48, groups=3),
            pool_func(kernel_size=3, stride=2, ceil_mode=True),
            fire_func(48, 64, groups=2),
            fire_func(64, 64, groups=2),
            fire_func(64, 96, groups=2),
            fire_func(96, 96, groups=3),
            pool_func(kernel_size=3, stride=2, ceil_mode=True),
            fire_func(96, 128, groups=2),
            fire_func(128, 128, groups=2),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data, a=-0.25)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data, gain=2.5)
                else:
                    init.normal(mod.weight.data)

    def forward(self, input):
        return self.conv_net.forward(input)


class TestSqueeze(nn.Module):
    def __init__(self, fire_func=Fire, pool_func=nn.AvgPool2d):
        super().__init__()
        self.conv_net = nn.Sequential(
            fire_func(32, 32, groups=2),
            fire_func(32, 48, groups=2),
            fire_func(48, 48, groups=3),
            pool_func(kernel_size=3, stride=2, ceil_mode=True),
            fire_func(48, 64, groups=2),
            fire_func(64, 64, groups=2),
            fire_func(64, 96, groups=2),
            fire_func(96, 96, groups=3),
            pool_func(kernel_size=3, stride=2, ceil_mode=True),
            fire_func(96, 128, groups=2),
            fire_func(128, 128, groups=2),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data, a=-0.25)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data, gain=2.5)
                else:
                    init.normal(mod.weight.data)

    def forward(self, input):
        return self.conv_net.forward(input)