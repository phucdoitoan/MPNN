

import torch
import torch.nn as nn


class UpdateFunction(nn.Module):

    def __init__(self, n_dim, m_dim):
        super(UpdateFunction, self).__init__()

        self.gru = torch.nn.GRU(input_size=m_dim, hidden_size=n_dim)

    def forward(self, h_v, m_v):
        """
        Update the feature vectors for all nodes in the large graph of a batch
        :param h_v: FloatTensor, n_node x n_dim
        :param m_v: FloatTensor, n_node x m_dim
        :return:
                FloatTensor, n_node x n_dim
        """

        h_v_new = self.gru(m_v.unsqueeze(0), h_v.unsqueeze(0))[0].squeeze()  # shape: n_node x n_dim

        return h_v_new

