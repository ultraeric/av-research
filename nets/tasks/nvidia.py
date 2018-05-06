import torch.nn as nn
import torch
import torch.nn.init as init
from utils.config import Config
from nets.subnets.module import Module
from torch.autograd import Variable
import nets.models.nvidia as nvidia

activation = nn.ELU


class E2ENet(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['constants']['embeddingDim']
        self.embedding_interp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.config['training']['outputFrames'] * 2),
        )

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data, a=-0.25)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data, gain=2.5)
                else:
                    init.normal(mod.weight.data)

    def forward(self, embedding):
        batch_size = embedding.size(0)
        return self.embedding_interp(embedding).contiguous().view(batch_size, -1, 2)


def unit_test():
    test_config = {'training': {'inputFrames': 5, 'outputFrames': 5}, 'constants': {'metadataDim': 16, 'embeddingDim': 64}}
    input = nvidia.unit_test(test_config)
    test_net = E2ENet(test_config)
    print(test_net(input).size())


unit_test()

Net = E2ENet

