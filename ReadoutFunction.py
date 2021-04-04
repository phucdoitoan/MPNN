

from .nnet import NNet

import torch
import torch.nn as nn


class ReadoutFunction(nn.Module):

    def __init__(self, n_dim, g_dim, nnet_hlayers_i=None, nnet_hlayers_j=None):
        """

        :param n_dim: dimension of feature vectors of the nodes
        :param g_dim: dimension of feature vectors of the graphs
        """
        super(ReadoutFunction, self).__init__()

        if nnet_hlayers_i is not None:
            self.i_net = NNet(n_in=2*n_dim, n_out=g_dim, hlayers=nnet_hlayers_i)
        else:
            self.i_net = NNet(n_in=2 * n_dim, n_out=g_dim)

        if nnet_hlayers_j is not None:
            self.j_net = NNet(n_in=n_dim, n_out=g_dim, hlayers=nnet_hlayers_j)
        else:
            self.j_net = NNet(n_in=n_dim, n_out=g_dim)

        self.n_dim = n_dim
        self.g_dim = g_dim

    def forward(self, h_T, h_0, graph_index):
        """
        readout for each individual small graphs in the batch
        n_node is the total number of nodes of all small graphs in the batch
        :param h_T: FloatTensor, n_node x n_dim
        :param h_0: FloatTensor, n_node x n_dim
        :param graph_index: LongTensor, n_node, indicate which small graph each node belongs to
        :return:
                FloatTensor, n_graph x g_dim, the feature vectors of each small graphs in the batch
                (n_graph = batch_size)
        """

        h_T0 = torch.cat([h_T, h_0], dim=1)  # shape: n_node x 2*n_dim

        i_h_TO = torch.sigmoid(self.i_net(h_T0))  # shape: n_node x g_dim
        j_h_T = self.j_net(h_T)  # shape: n_node x g_dim

        R_v = i_h_TO * j_h_T  # shape: n_node x g_dim

        # readout for each individual small graph
        n_graph = graph_index.max().item() + 1
        R = torch.zeros((n_graph, self.g_dim), device=h_0.device)
        R.index_add_(0, graph_index, R_v)  # shape: n_graph x g_dim

        return R
