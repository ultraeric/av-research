"""
Script to train neural networks.
"""
import importlib
import torch.nn.utils as nnutils
import torch.optim as optim
import torch
import utils
from utils.logging import *
from torch.autograd import Variable
from objects import Dataset
from nets.subnets.module import Module
from objects.task import TaskManager

config = utils.config.Config()
task_configs = [config['tasks'][task_name] for task_name in config['training']['tasks'].keys()]
torch.cuda.set_device(config['hardware']['gpu'])

Net = Module
Net = importlib.import_module(config['model']['pyPath']).Net
tasks = [importlib.import_module(task_config['pyPath']).Task(task_config['name'], config) for task_config in task_configs]
for task in tasks:
    task.cuda()
task_manager = TaskManager(tasks)
console_logger = ConsoleLogger()


def print_outputs(arr):
    if type(arr) != list and type(arr) != tuple:
        print(arr)
    else:
        str_list = ['{}'.format(_) for _ in arr]
        print(', '.join(str_list))


def iterate(net, task, optimizer=None, input=None, truth=None, train=True, i=-1):
    """
    Encapsulates a training or validation iteration.

    :param net: <nn.Module>: network to train
    :param loss_func: <nn.Module>: a module to calculate the loss. Typically from torch.nn
    :param optimizer: <torch.optim>: optimizer to use
    :param input: <tuple>: tuple of np.array or tensors to pass into net. Should contain data for this iteration
    :param truth: <np.array | tensor>: tuple of np.array to pass into optimizer. Should contain data for this iteration
    :param train: <bool>: True for training mode, False for eval mode
    :return: <np.float>: loss
    """
    if train:
        net.train()
        task.net.train()
        optimizer.zero_grad()
        task.optimizer.zero_grad()
    else:
        net.eval()
        task.net.eval()

    input = tuple([tensor if isinstance(tensor, torch.FloatTensor) else tensor for tensor in input])
    input = tuple([Variable(tensor).cuda() for tensor in input])
    embeddings = net(*input)
    outputs = task.net(embeddings)
    truth = Variable(truth).cuda()
    loss = task.loss_func(outputs, truth)

    if i % 25 == 0:
        print_outputs(cuda_to_list(outputs)[0])
        print_outputs(cuda_to_list(truth)[0])

    if train:
        loss.backward()
        if i % 25 == 0:
            print(list(net.parameters())[0].grad[0][0])
            print(list(net.parameters())[-1].grad[0])
            print(list(task.net.parameters())[-1].grad[0])
        nnutils.clip_grad_norm(net.parameters(), 25.)
        nnutils.clip_grad_norm(task.net.parameters(), 25.)

        task.optimizer.step()
        optimizer.step()

    return loss.cpu().data[0]


def main():
    # Basic training and network parameters
    net = Net(config=config).cuda()

    # Logic for resuming training
    if config['training']['startEpoch'] != 0:
        start_epoch = config['training']['startEpoch']-1
        net.load(config['model']['saveDirPath'], config['model']['name'], 'core', start_epoch)
        task_manager.load_nets(start_epoch)

    optimizer = optim.Adam(net.parameters(), lr=config['training']['learningRate'] or 1e-3, eps=1e-4, weight_decay=1e-4)

    for epoch in range(config['training']['startEpoch'], config['training']['numEpochs']):
        task_manager.init_epoch(epoch)
        train_loader = task_manager.get_train_loader()

        train_logger = LossLogger(train=True, config=config)

        # Training loop
        for i, (input, metadata, truth, task) in enumerate(train_loader):
            input = [input, metadata]
            loss = iterate(net, task, optimizer=optimizer, input=input, truth=truth, i=i)

            train_logger.store(loss)
            console_logger.loss(True, epoch, i, task_manager.curr_train_batch_len, loss,
                                debounce_buffer=2)
        train_logger.flush()

        val_loader = task_manager.get_val_loader()
        val_logger = LossLogger(train=False, config=config)

        # Validation loop
        for i, (input, metadata, truth, task) in enumerate(val_loader):
            input = [input, metadata]
            loss = iterate(net, task, input=input, truth=truth, train=False, i=i)
            val_logger.store(loss)
            console_logger.loss(False, epoch, i, task_manager.curr_val_batch_len, loss,
                                debounce_buffer=2)
        val_logger.flush()
        net.save(config['model']['saveDirPath'], config['model']['name'], 'core', epoch)
        task_manager.save_nets(epoch)


main()
