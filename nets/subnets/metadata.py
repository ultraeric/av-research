import torch.nn as nn
import torch
import torch.nn.init as init

class Metadata(nn.Module):
    """
    This class implements metadata encoding/decoding.
    """
    def __init__(self, metadata_dim=8, embedding_dim=32):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.embedding_dim = embedding_dim

        self.lookup = nn.Embedding(self.metadata_dim, self.embedding_dim)

        for mod in self.modules():
            if hasattr(mod, 'weight') and hasattr(mod.weight, 'data'):
                if isinstance(mod, nn.Conv2d):
                    init.kaiming_normal(mod.weight.data, a=-0.25)
                elif len(mod.weight.data.size()) >= 2:
                    init.xavier_normal(mod.weight.data, gain=2.5)
                else:
                    init.normal(mod.weight.data)

    def forward(self, input):
        """

        :param input: <LongTensor> of <batch_size, num_embeddings>
        :return: Embedded metadata vector
        """
        return self.lookup(input)


def unit_test():
    test_net = Metadata()
    test_net.forward(torch.LongTensor([[1], [2]]))

