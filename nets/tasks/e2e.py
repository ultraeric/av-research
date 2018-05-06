import torch.nn as nn
import torch
import torch.nn.init as init
from utils.config import Config
from nets.subnets.module import Module
import nets.models.frfcn as frfcn

activation = nn.ELU


class E2ENet(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding_dim = config['constants']['embeddingDim']
        self.decoder = nn.LSTM(1, 64, 1, batch_first=True)
        self.embedding_interp = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(self.embedding_dim, 24),
            activation(inplace=True),
            nn.BatchNorm1d(24),
            nn.Linear(24, 24),
            activation(inplace=True),
            nn.BatchNorm1d(24),
            nn.Linear(24, 24),
            activation(inplace=True),
            nn.Linear(24, 24),
            activation(inplace=True),
            nn.Linear(24, 2),
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
        embedding = [embedding[0], embedding[1]]
        batch_size = embedding[0].size(0)
        embedding[0] = embedding[0].view([1, embedding[0].size(0), embedding[0].size(1)])
        input, (h0, c0) = self.decoder(self.get_decoder_inputs(batch_size).cuda(), embedding)
        input = input.contiguous().view(batch_size * self.config['training']['outputFrames'], -1)
        return self.embedding_interp(input).contiguous().view(batch_size, -1)

    def get_decoder_inputs(self, batch_size=1):
        return torch.zeros([batch_size, self.config['training']['outputFrames'], 1])


def unit_test():
    test_config = {'training': {'inputFrames': 5, 'outputFrames': 5}, 'constants': {'metadataDim': 16, 'embeddingDim': 64}}
    input = frfcn.unit_test(test_config)
    test_net = E2ENet(test_config)
    test_net(input)


#unit_test()

Net = E2ENet

