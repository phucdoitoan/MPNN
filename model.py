

import torch
import torch.nn as nn

from .nnet import NNet
from .MessageFunction import MessageFunction
from .UpdateFunction import UpdateFunction
from .ReadoutFunction import ReadoutFunction


class MPNN_GraphClassifer(nn.Module):

    def __init__(self, n_dim, e_dim, m_dim, g_dim, T=3, m_nnet_hlayers=None, r_nnet_hlayers_i=None, r_nnet_hlayers_j=None, class_nnet_hlayers=None):
        super(MPNN_GraphClassifer, self).__init__()

        self.m_function = MessageFunction(n_dim=n_dim, e_dim=e_dim, m_dim=m_dim,
                                          nnet_hlayers=m_nnet_hlayers)
        self.u_function = UpdateFunction(n_dim=n_dim, m_dim=m_dim)
        self.r_function = ReadoutFunction(n_dim=n_dim, g_dim=g_dim,
                                          nnet_hlayers_i=r_nnet_hlayers_i,
                                          nnet_hlayers_j=r_nnet_hlayers_j)
        
        if class_nnet_hlayers is not None:
            self.class_net = NNet(n_in=g_dim, n_out=1, hlayers=class_nnet_hlayers)
        else:
            self.class_net = NNet(n_in=g_dim, n_out=1, hlayers=(128,))

        self.T = T  # number of layers

    def forward(self, h_0, graph_index, E, E_attr):
        """

        :param h_0: FloatTensor, n_node x n_dim: initial feature matrix of the nodes
        :param graph_index: LongTensor, n_node: indicates which node belong to which graph
        :param E: LongTensor, 2 x n_edge: edges
        :param E_attr: FloatTensor, e_dm x n_edge: attributes of edges
        :return:
        """
        n_node = h_0.shape[0]  # number of nodes
        h_v = h_0.clone()  # initiate h_v as new tensor having the same data as h_0

        # message passing and node feature updating
        for i in range(self.T):
            m_v = self.m_function(index_v=E[0], h_w=h_v[E[1]], e_vw=E_attr.t(), n_node=n_node)  # shape: n_node x m_dim
            h_v = self.u_function(h_v=h_v, m_v=m_v)  # shape: n_node x n_dim

        # readout
        R = self.r_function(h_v, h_0, graph_index)  # n_graph (batch_size) x g_dim

        # predict
        pred = self.class_net(R).squeeze()  # shape: n_graph
        pred = torch.sigmoid(pred)

        return pred






