

import torch
from  torch.utils.data import random_split

from model import MPNN_GraphClassifer
from utils import load_data, MPNN_Dataset, MPNN_DataLoader

import time
from sklearn.metrics import roc_auc_score
import numpy as np


BATCH_SIZE = 300 #50
LR = 0.001

n_DIM = 37
m_DIM = 37
e_DIM = 37*2
g_DIM = 8 #16

EPOCHS = 100 # 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

data = load_data('data/NCI1/NCI1.txt', n_labels=37)
print('len data: ', len(data))
dataset = MPNN_Dataset(data)

train_ratio = 0.9
train_dataset, test_dataset = random_split(dataset, lengths=(int(train_ratio*len(data)), len(data) - int(train_ratio*len(data))))

train_dataloader = MPNN_DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = MPNN_DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = MPNN_GraphClassifer(n_dim=n_DIM, e_dim=e_DIM, m_dim=m_DIM, g_dim=g_DIM,
                            m_nnet_hlayers=(8, 8),
                            r_nnet_hlayers_i=(8, 4),
                            r_nnet_hlayers_j=(8, 4),
                            class_nnet_hlayers=(8, 4))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = torch.nn.BCELoss()

loss_list = []

# train
for epoch in range(EPOCHS):

    t0 = time.time()
    correct = 0
    batch_len = 0

    pred_list = []
    label_list = []

    for h_0, graph_index, E, E_attr, labels in train_dataloader:
        h_0, graph_index, E, E_attr, labels = \
            h_0.to(device), graph_index.to(device), E.to(device), E_attr.to(device), labels.to(device)

        optimizer.zero_grad()

        preds = model(h_0, graph_index, E, E_attr)

        loss = criterion(preds, labels)
        loss.backward()

        loss_list.append(loss.item())

        optimizer.step()

        label_pred = (preds.detach() > 0.5).long()

        correct += (label_pred == labels).sum().item()
        batch_len += len(labels)

        pred_list.extend(preds.detach().tolist())
        label_list.extend(labels.tolist())

    train_acc = correct / batch_len
    train_loss = torch.tensor(loss_list).mean()

    auc = roc_auc_score(np.array(label_list), np.array(pred_list))

    print('Epoch %s: Train Loss: %.5f - Train Accuracy: %.2f - Train AUC: %.2f - Time: %.2fs'
          % (epoch, train_loss, train_acc, auc, time.time() - t0))


# test
loss_list = []
with torch.no_grad():
    t0 = time.time()
    correct = 0
    batch_len = 0

    pred_list = []
    label_list = []

    for h_0, graph_index, E, E_attr, labels in test_dataloader:
        h_0, graph_index, E, E_attr, labels = h_0.to(device), graph_index.to(device), E.to(device), E_attr.to(
            device), labels.to(device)
        preds = model(h_0, graph_index, E, E_attr)
        loss = criterion(preds, labels)

        loss_list.append(loss.item())

        label_pred = (preds > 0.5).long()

        correct += (label_pred == labels).sum().item()
        batch_len += len(labels)

        pred_list.extend(preds.tolist())
        label_list.extend(labels.tolist())

    test_acc = correct / batch_len
    test_loss = torch.tensor(loss_list).mean()

    auc = roc_auc_score(np.array(label_list), np.array(pred_list))

    print('Test! Test data length: ', len(label_list))

    print('Test Loss: %.5f - Test Accuracy: %.2f - Test AUC: %.2f - Time: %.2fs'
          % (test_loss, test_acc, auc, time.time() - t0))







