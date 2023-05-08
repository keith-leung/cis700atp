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
import main_graph_class
import main_node_class


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ''' omit this 
    result1 = main_graph_class.train_and_test(200)

    result11 = result1[0]
    result12 = result1[1]
    result13 = result1[2]

    train_loss = []
    train_accu = []
    loss1 = []
    loss2 = []
    loss3 = []
    train_accu1 = []
    train_accu2 = []
    train_accu3 = []
    test_accu1 = []
    test_accu2 = []
    test_accu3 = []
    for item in result11:
        loss1.append(item[1])
        train_accu1.append(item[2])
        test_accu1.append(item[3])
    for item in result12:
        loss2.append(item[1])
        train_accu2.append(item[2])
        test_accu2.append(item[3])
    for item in result13:
        loss3.append(item[1])
        train_accu3.append(item[2])
        test_accu3.append(item[3])
    train_loss = np.array([loss1, loss2, loss3])
    train_accu = np.array([train_accu1, train_accu2, train_accu3])
    test_accu = np.array([test_accu1, test_accu2, test_accu3])

    pt.subplot(3, 1, 1)
    pt.plot(train_loss.T)
    pt.xlabel("Iteration")
    pt.ylabel("Training loss")
    pt.title("Learning curves (%d reps)" % num_reps)
    pt.subplot(3, 1, 2)
    pt.plot(train_accu.T)
    pt.xlabel("Iteration")
    pt.ylabel("Train accuracy")
    pt.title("Train accuracy (%d reps)" % num_reps)
    pt.subplot(3, 1, 3)
    pt.plot(test_accu.T)
    pt.xlabel("Iteration")
    pt.ylabel("Test accuracy")
    pt.title("Test accuracy (%d reps)" % num_reps)
    pt.tight_layout()
    # pt.savefig(model.__name__ + ".png")
    pt.show()

    input('node classification results.....')
    '''

    num_reps = 3

    result2 = main_node_class.train_and_test(10)
    result21 = result2[0]
    result22 = result2[1]
    result23 = result2[2]


    train_loss = []
    train_accu = []
    loss1 = []
    loss2 = []
    loss3 = []
    train_accu1 = []
    train_accu2 = []
    train_accu3 = []
    test_accu1 = []
    test_accu2 = []
    test_accu3 = []
    for item in result21:
        loss1.append(item[1])
        train_accu1.append(item[2])
        test_accu1.append(item[3])
    for item in result22:
        loss2.append(item[1])
        train_accu2.append(item[2])
        test_accu2.append(item[3])
    for item in result23:
        loss3.append(item[1])
        train_accu3.append(item[2])
        test_accu3.append(item[3])
    train_loss = np.array([loss1, loss2, loss3])
    train_accu = np.array([train_accu1, train_accu2, train_accu3])
    test_accu = np.array([test_accu1, test_accu2, test_accu3])

    pt.subplot(3, 1, 1)
    pt.plot(train_loss.T)
    pt.xlabel("Iteration")
    pt.ylabel("Training loss")
    pt.title("Learning curves (%d reps)" % num_reps)
    pt.subplot(3, 1, 2)
    pt.plot(train_accu.T)
    pt.xlabel("Iteration")
    pt.ylabel("Train accuracy")
    pt.title("Train accuracy (%d reps)" % num_reps)
    pt.subplot(3, 1, 3)
    pt.plot(test_accu.T)
    pt.xlabel("Iteration")
    pt.ylabel("Test accuracy")
    pt.title("Test accuracy (%d reps)" % num_reps)
    pt.tight_layout()
    # pt.savefig(model.__name__ + ".png")
    pt.show()

    input('all results.....')


