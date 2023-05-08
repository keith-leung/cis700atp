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
device_laptop = tr.device("cuda" if tr.cuda.is_available() else "cpu")


"""
very simple Graph Neural Network
"""
class SimpleGNN(tr.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = tr.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train_and_test(total_epoch):
    list_results = []

    dataset = get_data_loader()

    def train(dataloader):
        model.train()
        total_loss = 0
        for data in dataloader:
            #print(data)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(dataloader.dataset)

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for k in range(3):
        import random
        random.shuffle(dataset)

        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]

        dataloader = DataLoader(train_dataset, 1, shuffle=True)
        testdataloader = DataLoader(test_dataset, 1, shuffle=False)

        input_dim = dataloader.dataset[0].num_node_features
        hidden_dim = 4
        model = SimpleGNN(input_dim, hidden_dim, 2)
        criterion = tr.nn.CrossEntropyLoss()
        optimizer = tr.optim.Adam(model.parameters(), lr=0.0005)
        list_one = []

        for epoch in range(1, total_epoch + 1):
            loss = train(dataloader)
            train_acc = test(dataloader)
            test_acc = test(testdataloader)
            list_one.append((epoch, loss, train_acc, test_acc))
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        list_results.append(list_one)

    return list_results

def get_data_loader():
    return data_loader_basic.get_data_loader(1, False)
    #return data_loader_basic.data_loader_clause()
    #return data_loader_basic.get_data_lemma()
