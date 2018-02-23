import torch
import torch.nn as nn
import torch.nn.init as init
from utils.config import Config
from torch.autograd import Variable
import sys
sys.path.append('../..')
import nets.subnets as subnets

activation = nn.ELU
pool_func = nn.AvgPool2d


# from Parameters import ARGS
def Fire(*args, **kwargs):
    kwargs['activation'] = activation
    kwargs['dropout'] = max(0.5 * (1 - (16. / args[1])), 0.)
    return subnets.Fire(*args, **kwargs)


class SqueezeNet(subnets.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.input_frames = config['constants']['inputFrames']
        self.metadata_dim = config['constants']['metadataDim']
        self.embedding_dim = config['constants']['embeddingDim']

        self.pre_squeeze = nn.Sequential(
            nn.Conv2d(3 * self.input_frames, 12, kernel_size=3, stride=1, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
        )
        self.squeeze_submodule = subnets.SqueezeSubmodule(fire_func=Fire, pool_func=pool_func)
        self.post_squeeze = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=2),
            activation(inplace=True),
            nn.Conv2d(128, 96, kernel_size=3, stride=2, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1),
            activation(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        )
        self.meta_embedding = subnets.Metadata(8, self.metadata_dim)
        self.post_metadata = nn.Sequential(
            nn.Linear(128 + self.metadata_dim, 144),
            activation(inplace=True),
            nn.BatchNorm1d(144),
            nn.Linear(144, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.embedding_dim),
            activation(inplace=True),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data, a=-0.25)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data, gain=2.5)
                else:
                    init.normal(mod.weight.data)

    def forward(self, input, meta_indices):
        input = self.format_frames(input, feat_dim=1)
        input = self.pre_squeeze(input)
        input = self.squeeze_submodule(input)
        input = self.post_squeeze(input)
        input = input.contiguous().view(input.size(0), -1)
        meta_embeddings = self.meta_embedding(meta_indices)
        meta_embeddings = meta_embeddings.contiguous().view(meta_indices.size(0), -1)
        meta_included = torch.cat([input, meta_embeddings], 1)
        post_metadata = self.post_metadata(meta_included)
        return post_metadata

    def num_params(self):
        num_params = 0
        for parameter in self.parameters():
            curr_params = 1
            for dimension in parameter.size():
                curr_params *= dimension
            num_params += curr_params
        return num_params


def unit_test():
    test_config = {'constants': {'inputFrames': 5, 'metadataDim': 16, 'embeddingDim': 64}}
    test_net = SqueezeNet(test_config)
    test_net_output = test_net(Variable(torch.randn(2, 5, 180, 320, 3)),
                      Variable(torch.LongTensor([[1], [2]])))
    print(test_net_output.size())
    print(test_net.num_params())


unit_test()

Net = SqueezeNet
