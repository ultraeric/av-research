import torch
import torch.nn as nn
import torch.nn.init as init
import sys
from utils.config import Config
sys.path.insert(0, '../../')
import nets.subnets as subnets


class Nvidia(subnets.Module):

    def __init__(self, config: Config):
        super(Nvidia, self).__init__()

        self.input_frames = config['training']['inputFrames']
        self.metadata_dim = config['constants']['metadataDim']
        self.embedding_dim = config['constants']['embeddingDim']
        self.conv_nets = nn.Sequential(
            nn.Conv2d(3 * self.input_frames, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(0.5),
            nn.Conv2d(48, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.fcl = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(p=0.5),
            nn.Linear(250, self.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(p=0.5),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data)
                else:
                    init.normal(mod.weight.data)


    def forward(self, x, metadata):
        x = self.format_frames(x, feat_dim=1)
        x = self.conv_nets(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        x = x.contiguous().view(x.size(0), -1)
        return x


def unit_test(test_config={}):
    test_config = test_config if test_config else {'training': {'inputFrames': 5}, 'constants': {'metadataDim': 16, 'embeddingDim': 64}}
    test_net = Nvidia(test_config)
    a = test_net(torch.randn(7, 5, 94, 168, 3),
                 torch.LongTensor([[1], [2], [3], [4], [5], [6], [7]]))
    print(a.size())
    return a

unit_test()

Net = Nvidia