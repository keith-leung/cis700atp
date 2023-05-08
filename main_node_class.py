import sys
import matplotlib.pyplot as pt
import numpy as np
import torch as tr
import torch.nn.functional as F
import random
import torch_geometric.nn
from torch_geometric.data import *
from torch_geometric.nn import *
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data, DataLoader
from utils import *
import data_loader_basic
import data_loader_lemma
import data_loader_clause

set_mm_path = 'set.mm'


class GCN(tr.nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super().__init__()
        self.gcn = GCNConv(num_features, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.out = tr.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gcn2(h, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        z = tr.nn.functional.softmax(self.out(h))
        return h, z

def train_and_test(total_epoch):

    list_results = []

    dataset, num_features = get_data_loader()

    # Calculate accuracy
    def accuracy(pred_y, y):
        return (pred_y == y).sum() / len(y)
        '''
        non_zero_indices = tr.nonzero(y)
        selected_columns1 = tr.index_select(pred_y, dim=0, index=non_zero_indices.squeeze())
        selected_columns2 = tr.index_select(y, dim=0, index=non_zero_indices.squeeze())
        return (selected_columns1 == selected_columns2).sum() / len(non_zero_indices)
        '''

    def train(dataloader):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for data in dataloader:
            h, z = model(data.x, data.edge_index)

            # Calculate loss function
            loss = criterion(z, data.y)

            # Calculate accuracy
            acc = accuracy(z.argmax(dim=1), data.y)

            losses.append(loss.item())
            total_loss += loss.item()
            # Compute gradients
            loss.backward()

            # Tune parameters
            optimizer.step()

            # Store data for animations
            embeddings.append(h)
            accuracies.append(acc)

        return total_loss / len(dataloader.dataset)

    def test(loader):
        model.eval()
        test_accuracies = []

        for data in loader:  # Iterate in batches over the training/test dataset.
            h, z = model(data.x, data.edge_index)
            acc2 = accuracy(z.argmax(dim=1), data.y)
            test_accuracies.append(acc2)

        result = sum(test_accuracies) / len(test_accuracies)
        return result  # Derive ratio of correct predictions.

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []


    for k in range(3):
        import random
        random.shuffle(dataset)

        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]

        dataloader = DataLoader(train_dataset, 1, shuffle=True)
        testdataloader = DataLoader(test_dataset, 1, shuffle=False)

        input_dim = dataloader.dataset[0].num_node_features

        criterion = tr.nn.CrossEntropyLoss()
        model = GCN(num_features=input_dim, hidden=3, num_classes=2)
        optimizer = tr.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
                                  #lr=0.00005, weight_decay=5e-4)
        list_one = []

        for epoch in range(1, 1 + total_epoch):
            loss = train(dataloader)
            train_acc = test(dataloader)
            test_acc = test(testdataloader)
            list_one.append((epoch, loss, train_acc, test_acc))
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        list_results.append(list_one)

    return list_results

def get_data_loader():
    return data_loader_lemma.get_data_loader(1, False)
