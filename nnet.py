

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NNet(nn.Module):
    """a small neural network with fully connected layers"""

    def __init__(self, n_in, n_out, hlayers=(128, 256, 128)):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList(
            [nn.Linear(n_in, hlayers[i]) if i == 0 else
             nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
             nn.Linear(hlayers[i-1], hlayers[i]) for i in range(self.n_hlayers+1)]
        )

        for model in self.fcs:
            init.xavier_uniform_(model.weight)

    def forward(self, x):
        """

        :param x: batch x n_in
        :return: batch x n_out
        """
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x

