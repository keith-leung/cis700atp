import traceback

from utils import *
from torch_geometric.data import Data, DataLoader

def get_data_loader(batch_size:int, shuffle:bool):
    thms = load_database(1000)

    set_lemmas = set([])
    length_proof = 0
    for label, (assrt, essen, proof) in thms.items():
        print((label, assrt, essen, proof))
        set_lemmas.add(label)
        length_current = 1
        for essen_item in essen:
            set_lemmas.add(essen_item[0])
        for proof_item in proof:
            set_lemmas.add(proof_item)
            length_current += 1
        length_proof = max(length_current, length_proof)

    num_nodes = len(set_lemmas)
    num_of_features = length_proof
    list_tokens = sorted(list(set_lemmas))

    char_to_idx = {char: idx for idx, char in enumerate(list_tokens)}
    print(char_to_idx)

    graphs = []
    for label, (assrt, essen, proof) in thms.items():
        # skip axioms, they do not have proofs
        if len(proof) == 0:
            continue

        list_nodes = []
        for token in list_tokens:
            if token == label:
                list_nodes.append([1])
            else:
                list_nodes.append([0])

        length = len(proof)
        for j in range(num_of_features):
            for i, token in enumerate(list_tokens):
                if j >= length - 1: #tailor the last edge
                    list_nodes[i].append(0)
                    continue

                prof = proof[j]
                if token == prof:
                    list_nodes[i].append(1)
                else:
                    list_nodes[i].append(0)

        x = tr.tensor(list_nodes, dtype=tr.float)

        edge_list = [list_tokens.index(label)]
        for i, prof in enumerate(proof):  # add all the edges
        # for i, prof in enumerate(proof[:-1]): #tailor the last edge
            edge_list.append(list_tokens.index(prof))

        edge_index = tr.tensor([edge_list[:-1], edge_list[1:]], dtype=tr.long)

        lst_y = []  # y has all the nodes' feature
        for i, token in enumerate(list_tokens):
            if token in proof or token == label:
                lst_y.append(1)
            else:
                lst_y.append(0)

        y = tr.tensor(lst_y).to(tr.long)
            #.randint(low=0, high=4, size=(1,)).to(tr.long)
            # tr.tensor([lst_y0, lst_y1], dtype=tr.float).t() # tr.tensor([1.0], dtype=tr.float)
        #datay = Data(x, y)
        #print(x)

        data = Data(x, edge_index, y = y)
        graphs.append(data)
        pass

    dataset = graphs[0:]
    return dataset, len(graphs[0].x[0])
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    #return loader