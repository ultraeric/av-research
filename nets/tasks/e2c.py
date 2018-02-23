import torch.nn as nn
import torch
import torch.nn.init as init
from utils.config import Config
from nets.subnets.module import Module

activation = nn.ELU


class E2CNet(Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding_dim = config['constants']['embeddingDim']

        self.embedding_interp = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(self.embedding_dim, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.Linear(128, 5),
            nn.Sigmoid()
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
        return self.embedding_interp(embedding)


Net = E2CNet
