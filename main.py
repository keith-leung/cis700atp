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

set_mm_path = 'C:/Users/Public/Documents/metamath/set.mm'
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


class GCN(tr.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gcn = GCNConv(num_features, 3)
        self.gcn2 = GCNConv(3, 3)
        self.out = tr.nn.Linear(3, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gcn2(h, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        z = tr.nn.functional.softmax(self.out(h))
        return h, z


def train_and_test():
    dataset, num_features = get_data_loader()
    model = GCN(num_features=num_features, num_classes=2)
    optimizer = tr.optim.Adam(model.parameters(),  lr=0.0005) # lr=0.00001)
    criterion = tr.nn.CrossEntropyLoss()

    import random
    random.shuffle(dataset)

    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]

    # Calculate accuracy
    def accuracy(pred_y, y):
        return (pred_y == y).sum() / len(y)
        '''
        non_zero_indices = tr.nonzero(y)
        selected_columns1 = tr.index_select(pred_y, dim=0, index=non_zero_indices.squeeze())
        selected_columns2 = tr.index_select(y, dim=0, index=non_zero_indices.squeeze())
        return (selected_columns1 == selected_columns2).sum() / len(non_zero_indices)
        '''

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    # Training loop
    for epoch in range(201):
        # Clear gradients
        optimizer.zero_grad()
        for data in train_dataset:
            # Forward pass
            h, z = model(data.x, data.edge_index)

            # Calculate loss function
            loss = criterion(z, data.y)

            # Calculate accuracy
            acc = accuracy(z.argmax(dim=1), data.y)

            # Compute gradients
            loss.backward()

            # Tune parameters
            optimizer.step()

            # Store data for animations
            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(z.argmax(dim=1))

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc * 100:.2f}%')

    model.eval()
    test_accuracies = []
    correct = 0
    for data in test_dataset:  # Iterate in batches over the training/test dataset.
        h, z = model(data.x, data.edge_index)
        acc2 = accuracy(z.argmax(dim=1), data.y)
        test_accuracies.append(acc2)
    result = sum(test_accuracies) / len(test_accuracies)  # Derive ratio of correct predictions.
    print(result)
    input("wait....")
    dataloader = DataLoader(train_dataset, 1, shuffle=True)
    testdataloader = DataLoader(test_dataset, 1, shuffle=False)

    input_dim = dataloader.dataset[0].num_node_features
    hidden_dim = 16
    model = GCN(num_features=input_dim, hidden_channels=hidden_dim)
    optimizer = tr.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(dataloader):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for data in dataloader:
            #print(data)
            out = model(data.x, data.edge_index)  # Perform a single forward pass.
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(dataloader.dataset)

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred == data.y  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(loader.dataset.__sizeof__())
            correct += test_acc
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(1, 101):
        loss = train(dataloader)
        train_acc = test(dataloader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')

    test_acc = test(testdataloader)

    print('acc: ' + str(test_acc))

   # model, train, losses, accs, test_accs = None
    return (1, 2, 3, 4, 5)

def show_and_the_rest(model, train, losses, accs, test_accs):
    if model is None:
        return

    # exponential moving averages
    ema_decay = 0.
    metrics = {"loss": losses, "acc": accs}
    emas = {}
    for name, metric in metrics.items():
        emas[name] = [metric[0]]
        for i in range(1, len(metric)):
            emas[name].append(ema_decay * emas[name][-1] + (1 - ema_decay) * metric[i])

    pt.subplot(3, 1, 1)
    pt.plot(emas['loss'])
    pt.ylabel("Training loss")
    pt.subplot(3, 1, 2)
    pt.plot(emas['acc'])
    pt.ylabel("Training accuracy")
    pt.xlabel("Update")
    pt.subplot(3, 1, 3)
    pt.plot(test_accs, label="Unbalanced")
    pt.ylabel("Testing accuracy")
    pt.xlabel("Epoch")
    pt.tight_layout()
    pt.show()

    num_samples = 3

    for i, (prompt, output) in enumerate(train):

        samples = []
        for s in range(num_samples):

            # initialize with prompt
            seq_so_far = list(prompt)
            seq_so_far += ["$="]  # tell it to start on the proof step
            seq_so_far += output[:2]  # some little hint
            for t in range(2, len(output)):

                with tr.no_grad():

                    # send example through model
                    logits = model(seq_so_far)

                    # sort probabilities for next prediction
                    probs = tr.softmax(logits[-1], dim=0)
                    sorted_probs, sorter = tr.sort(probs, descending=True)
                    sorted_probs = sorted_probs.cpu().numpy()

                    # extract top 95% of probability mass
                    keep = (sorted_probs.cumsum() > 0.95).argmax() + 1
                    probs = sorted_probs[:keep] / sorted_probs[:keep].sum()  # renormalize
                    choices = model.embedder.get_tokens(sorter[:keep])

                    # make first sample deterministic greedy, others random
                    if s == 0:
                        next_token = choices[0]
                    else:
                        next_token = np.random.choice(choices, p=probs)

                    seq_so_far.append(next_token)

            samples.append(seq_so_far)

        print("INPUT:")
        print(" ".join(prompt))
        print("\nTARGET:")
        print(" ".join(output))
        print("\nCONCAT:")
        print(" ".join(prompt + output))
        print("\nSAMPLES:")
        for sample in samples:
            print(" ".join(sample[len(prompt):]))

        input('\n...')

def get_data_loader():
    #return data_loader_basic.get_data_loader(1, False)
    #return data_loader_clause.get_data_loader()
    return data_loader_lemma.get_data_loader(1, False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model, train, losses, accs, test_accs = train_and_test()
   # DataLoader = get_data_loader()

    input("stucking for check ......")
    sys.exit(0)

    show_and_the_rest(model, train, losses, accs, test_accs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
