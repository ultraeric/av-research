"""SqueezeNet 1.1 modified for LSTM regression."""
import torch
import torch.nn as nn
import torch.nn.init as init
from utils.config import Config
import sys
sys.path.insert(0, '../../')
import nets.subnets as subnets

activation = nn.ELU


# from Parameters import ARGS
def Fire(*args, **kwargs):
    kwargs['activation'] = activation
    kwargs['dropout'] = max(0.5 * (1 - (16. / args[1])), 0.)
    return subnets.Fire(*args, **kwargs)


pool_func = nn.AvgPool2d


class SqueezeNetTimeLSTM(subnets.Module):
    """SqueezeNet+LSTM for end to end autonomous driving"""

    def __init__(self, config: Config):
        """Sets up layers"""
        super().__init__()

        self.input_frames = config['training']['inputFrames']
        self.metadata_dim = config['constants']['metadataDim']
        self.embedding_dim = config['constants']['embeddingDim']

        self.pre_squeeze = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            activation(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            activation(inplace=True),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(0.2)
        )
        self.squeeze_submodule = subnets.SqueezeSubmodule(fire_func=Fire, pool_func=pool_func)
        self.post_squeeze = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            activation(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=0, groups=1),
            activation(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=1),
            activation(inplace=True),
        )
        self.lstm_encoder = nn.ModuleList([
            nn.LSTM(32, 64, 1, batch_first=True)
        ])
        self.meta_embedding = subnets.Metadata(8, self.metadata_dim)
        self.post_metadata = nn.Sequential(
            nn.Linear(64 + self.metadata_dim, 64),
            activation(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            activation(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            activation(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.embedding_dim),
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
        """Forward-propagates data through SqueezeNetTimeLSTM submodule"""
        batch_size = input.size(0)
        input = self.format_frames(input, time_dim=1, feat_dim=2)
        input = input.contiguous().view(-1, *(input.size()[2:]))
        input = self.pre_squeeze(input)
        input = self.squeeze_submodule(input)
        input = self.post_squeeze(input)
        input = input.contiguous().view(batch_size, -1, input.size(1) * input.size(2) * input.size(3))
        for lstm in self.lstm_encoder:
            lstm_output, last_hidden_cell = lstm(input)
            input, output = last_hidden_cell[0], last_hidden_cell[1]
        post_lstm = input.contiguous().view(input.size(1), -1)
        meta_embeddings = self.meta_embedding(meta_indices)
        meta_embeddings = meta_embeddings.contiguous().view(meta_indices.size(0), -1)
        meta_included = torch.cat([post_lstm, meta_embeddings], 1)
        input = self.post_metadata(meta_included)
        return input, output

    def num_params(self):
        num_params = 0
        for parameter in self.parameters():
            curr_params = 1
            for dimension in parameter.size():
                curr_params *= dimension
            num_params += curr_params
        return num_params


def unit_test(test_config={}):
    """Tests SqueezeNetTimeLSTM for size constitency"""
    test_config = test_config if test_config else {'training': {'inputFrames': 5}, 'constants': {'metadataDim': 16, 'embeddingDim': 64}}
    test_net = SqueezeNetTimeLSTM(test_config)
    test_net_output = test_net(torch.randn(7, 5, 94, 168, 6),
                               torch.LongTensor([[1], [2], [3], [4], [5], [6], [7]]))
    return test_net_output


unit_test()

Net = SqueezeNetTimeLSTM
