import torch.nn as nn
import torch


def copy_in_params(net: nn.Module, params):
    """
    Copies parameters from a FP32 parameter copy to an FP16 model

    :param net:
    :param params:
    :return:
    """
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, net: nn.Module):
    """
    Copies parameters from an FP16 model to a FP32 parameter copy.

    :param params:
    :param net:
    :return:
    """
    for param, param_w_grad in zip(params, net.parameters()):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)
