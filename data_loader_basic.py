import traceback

from utils import *
from torch_geometric.data import Data, DataLoader

def get_data_loader(batch_size:int, shuffle:bool):
    thms = load_database(1880)

    set_tokens = set([])

    for label, (assrt, essen, proof) in thms.items():
        #print((label, assrt, essen, proof))
        prompt = tokenize(assrt, essen)
        set_tokens = set_tokens.union(set(prompt))
        # skip axioms, they do not have proofs
        if len(proof) == 0:
            continue
        proof_assrt, proof_essen, _ = thms[proof[-1]]
        output = tokenize(proof_assrt, proof_essen)
        set_tokens = set_tokens.union(set(output))

    num_nodes = len(set_tokens)
    num_of_features = 1
    list_tokens = sorted(list(set_tokens), reverse=True)

    char_to_idx = {char: idx for idx, char in enumerate(list_tokens)}
    #print(char_to_idx)

    graphs = []
    for label, (assrt, essen, proof) in thms.items():
        # skip axioms, they do not have proofs
        if len(proof) == 0:
            continue
        #key: label
        #value: assrt, essen, proof
        #print((label, assrt, essen, proof))
        prompt = tokenize(assrt, essen)

        input_indices = [char_to_idx[char] for char in prompt]
        #print(input_indices)
        x_indices = [1 if i in input_indices else 0 for i in range(num_nodes)]
        x = tr.tensor([x_indices], dtype=tr.float).t()
            # tr.randn(num_nodes, num_of_features, dtype=tr.float)
            #tr.tensor(x_indices, dtype=tr.float)
        #print(x)

        proof_assrt, proof_essen, _ = thms[proof[-1]]

        output = tokenize(proof_assrt, proof_essen)
        output_indices = [char_to_idx[char] for char in output]
        #print(output_indices)
        lst_x0 = input_indices[:-1]
        lst_x1 = input_indices[1:]
        lst_y0 = output_indices[:-1]
        lst_y1 = output_indices[1:]

        lst_x0.extend(lst_y0)
        lst_x1.extend(lst_y1)

        edge_index = tr.tensor([
            lst_x0,
            lst_x1
        ], dtype=tr.long)  # Edge indices

        y = tr.tensor([1]).to(tr.long)
            #.randint(low=0, high=4, size=(1,)).to(tr.long)
            # tr.tensor([lst_y0, lst_y1], dtype=tr.float).t() # tr.tensor([1.0], dtype=tr.float)
        #datay = Data(x, y)
        #print(x)

        data = Data(x, edge_index, y = y)
        graphs.append(data)

        #sample a random step not true
        pr_false = list(thms.items())[random.randint(0, len(thms.keys())-1)]
        proof_assrt, proof_essen, _ = thms[pr_false[0]]
        output = tokenize(proof_assrt, proof_essen)
        output_indices = [char_to_idx[char] for char in output]
        # print(output_indices)
        lst_x0 = input_indices[:-1]
        lst_x1 = input_indices[1:]
        lst_y0 = output_indices[:-1]
        lst_y1 = output_indices[1:]

        lst_x0.extend(lst_y0)
        lst_x1.extend(lst_y1)

        edge_index = tr.tensor([
            lst_x0,
            lst_x1
        ], dtype=tr.long)  # Edge indices

        y = tr.tensor([0]).to(tr.long)
        # .randint(low=0, high=4, size=(1,)).to(tr.long)
        # tr.tensor([lst_y0, lst_y1], dtype=tr.float).t() # tr.tensor([1.0], dtype=tr.float)
        # datay = Data(x, y)
        # print(x)

        data = Data(x, edge_index, y=y)
        graphs.append(data)
        pass

    dataset = graphs[0:]
    return dataset
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    #return loader