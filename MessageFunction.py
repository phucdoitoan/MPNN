

from .nnet import NNet

import torch
import torch.nn as nn


"""
The dataset contains of many small size graphs, which will be processed in batches of e.g: 20 graphs/batch
Each batch will be seen as a whole large graph consisting of many unconnected small components (small isolated subgraphs)
"""
class MessageFunction(nn.Module):

    def __init__(self, n_dim, e_dim, m_dim, nnet_hlayers=None):
        """

        :param n_dim: dimention of feature vector of a node
        :param e_dim: dimension of feature vector of an edge
        :param m_dim: dimension of feature vector of a message
        """
        super(MessageFunction, self).__init__()

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.m_dim = m_dim
        if nnet_hlayers is not None:
            self.m_function = NNet(e_dim, n_dim*m_dim, nnet_hlayers)  # learnable parameters
        else:
            self.m_function = NNet(e_dim, n_dim * m_dim)  # learnable parameters

    def forward(self, index_v, h_w, e_vw, n_node):
        """
        Perform message passing for all edges in the "large graph" of a batch
        If a sparse representation of the adjeciency matrix is written as
            edges: shape 2 x n_edge
            edges_attr: shape e_dim x n_edge
        , then it is assumed that
            index_v = edges[0]
            h_w is node feature vectors indexing with edges[1]
            e_vw = edges_attr.t()
        Note: it is supposed that  node_edge = adj.sum(dim=1)
        :param index_v: LongTensor, n_edge, target nodes (not used by the Edge Network), n_e is the number of edge of the "large graph" in a batch
        :param h_w: FloatTensor, n_edge x n_dim, source nodes
        :param e_vw: FloatTensor, n_edge x e_dim, edges that connect source nodes to target nodes
        :param n_node: int, number of nodes in the large graph
        :return:
                n_node x m_dim: messages to all nodes in the large graph
        """

        # matrix A for each edge
        A_vw = self.m_function(e_vw).view(-1, self.m_dim, self.n_dim)  # shape: n_edge x m_dim x n_dim

        # message for each edge
        m_vw = torch.bmm(A_vw, h_w.unsqueeze(-1)).squeeze()  # shape: n_edge x m_dim

        # summing up the message for each node
        #n_node = index_v.max().item() + 1
        #could not use the above to compute the number of nodes (as sometimes, there isolated nodes in the small graphs, e.g: molecular CCCCCCCCC=CCCCCCCCC(=O)[O-].[NH3+]CCO
        m_v = torch.zeros((n_node, self.m_dim), device=m_vw.device)
        m_v.index_add_(0, index_v, m_vw)  # shape: n_node x m_dim   # checked: seems to work fine

        return m_v


if __name__ == '__main__':
    messager = MessageFunction(100, 100, 100)

    for p in messager.parameters():
        print(type(p), p.shape)
