

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import networkx as nx
import numpy as np


def load_data(file_name='data/mutag/MUTAG.txt', n_labels=7, n_dim=None):
    data = []
    with open(file_name) as f:
        n_graphs = int(f.readline())
        for _ in range(n_graphs):
            n, y = map(int, f.readline().split())
            G = nx.Graph()
            G.add_nodes_from(range(n))
            node_labels = []
            for i in range(n):
                li = list(map(int, f.readline().split()))
                node_labels.append(int(li[0]))
                for j in li[2:]:
                    G.add_edge(i, j)
            if n_dim is not None:
                X = torch.zeros((len(node_labels), n_dim))
                X[:, :n_labels] = torch.eye(n_labels)[node_labels]
            else:
                X = torch.eye(n_labels)[node_labels]
            adj = nx.adjacency_matrix(G).tocoo()
            E = torch.LongTensor(np.vstack((adj.row, adj.col)))
            E_attr = torch.FloatTensor(torch.cat((X[E[0]], X[E[1]]), dim=1)).t()
            data.append((X, n, E, E_attr, int(y != 0)))

    return data


def adj_to_torch_sparse(G):
    coo_A = nx.adjacency_matrix(G).tocoo()
    i = torch.LongTensor()



class MPNN_Dataset(Dataset):

    def __init__(self, data):
        """

        :param data: iterables of tuples of form (X, n_node, E, E_attr, label)

        X: node feature matrix, FloatTensor: n_node x n_dim
        n_node: number of nodes, int
        E: edges, LongTensor, 2 x n_edge  (sparse representation of adjcency matrix)
        E_attr: edge feature matrix, FloatTensor, e_dim x n_edge
        label: 0 or 1
        """
        super(MPNN_Dataset, self).__init__()

        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def mpnn_collate_fn(batch):
    X = torch.cat([t[0] for t in batch], dim=0)  # concat feature matrix of all graphs
    node_nums = [t[1] for t in batch]  # list of num_node of each graph in the batch

    # edges of all graphs, node indices being offset
    E = torch.cat([t[2] + sum(node_nums[:i]) for i, t in enumerate(batch)], dim=1)

    E_attr = torch.cat([t[3] for t in batch], dim=1)  # concate edge feature matrix of all graphs
    labels = torch.FloatTensor([t[-1] for t in batch])  # labels of graphs in the batch

    # denote which node belongs to which graph (after being offset)
    graph_index = torch.LongTensor([item for i in range(len(node_nums)) for item in [i]*node_nums[i]])

    return X, graph_index, E, E_attr, labels


class MPNN_DataLoader(DataLoader):

    def __init__(self, dataset, **kwargs):
        super(MPNN_DataLoader, self).__init__(dataset=dataset, collate_fn=mpnn_collate_fn, **kwargs)


# check collate_fn
class ToyDataset(Dataset):

    def __init__(self, n=5):
        super(ToyDataset, self).__init__()

        self.n = n

    def __getitem__(self, item):
        s1 = torch.randint(1, self.n, size=(1,)).item()
        s2 = torch.randint(1, self.n, size=(1,)).item()
        return s1, s2, torch.randint(0, self.n, size=(s1, s2))

    def __len__(self):
        return 100


def toy_collate_fn(batch):
    l = torch.cat([t[-1].view(-1) for t in batch], dim=0)
    s1 = torch.LongTensor([t[0] for t in batch])
    s2 = torch.FloatTensor([t[1] for t in batch])
    return s1, s2, l


class Toy_DataLoader(DataLoader):
    def __init__(self, **kwargs):
        super(Toy_DataLoader, self).__init__(collate_fn=toy_collate_fn, **kwargs)


if __name__ == '__main__':
    toy_dataset = ToyDataset()
    toy_dataloader = Toy_DataLoader(dataset=toy_dataset, batch_size=5)
    for s1, s2, l in toy_dataloader:
        print('*************len l: ', len(l), int((s1*s2).sum().item()))
        print(l)

        print('len s1: ', len(s1))
        print(s1)
        print('len s2: ', len(s2))
        print(s2)

    mutag_data = load_mutag()
    mutag_dataset = MPNN_Dataset(mutag_data)
    mutag_dataloader = MPNN_DataLoader(mutag_dataset, batch_size=3)

    X, graph_index, E, E_attr, label = next(iter(mutag_dataloader))
    print('X: ', X.shape)
    print('graph_index: ', graph_index.shape, graph_index)
    print('E: ', E.shape)
    print('E: ', E[:, :54])
    print('E: ', E[:, 54:110])
    print('E: ', E[:, 110:])
    print('E_attr: ', E_attr.shape)
    #print('E_attr: ', E_attr[:, 0], '\n', E_attr[:, 2])
    print('label: ', label)



