import sys
import matplotlib.pyplot as pt
import numpy as np
import torch as tr
import random
from utils import *
import data_loader_basic
import data_loader_lemma
import data_loader_clause
from torch_geometric.data import Data, DataLoader

set_mm_path = 'C:/Users/Public/Documents/metamath/set.mm'
device_laptop = tr.device("cuda" if tr.cuda.is_available() else "cpu")


"""
built-in pytorch embedding module refers to tokens by index in vocabulary
this wrapper looks up by token string
"""

class Embedder(tr.nn.Module):
    def __init__(self, vocab, d_model, device):
        super(Embedder, self).__init__()

        # lookup table mapping tokens to indices
        self.vocab = tuple(sorted(vocab.keys()))
        self.lookup = {tok: t for t, tok in enumerate(self.vocab)}

        # torch embedding module
        self.embedding = tr.nn.Embedding(len(vocab), d_model, device=device)

        # remember device for transfering input
        self.device = device

    def get_index(self, tokens):
        return tr.tensor([self.lookup[tok] for tok in tokens], device=self.device)

    def get_tokens(self, index):
        return [self.vocab[i] for i in index]

    def forward(self, tokens):
        # lookup the token indices
        idx = self.get_index(tokens)
        # retrieve the embeddings
        return self.embedding(idx)



"""
Define seq2seq transformer
"""

class Net(tr.nn.Module):
    def __init__(self, vocab, max_len, denom_base, d_model, nhead, nlayers, ff_dim, device):
        super(Net, self).__init__()

        # Token embeddings
        self.embedder = Embedder(vocab, d_model, device)

        # Position encoding
        self.positional_encoder = trigonometric_positional_encoder(max_len, d_model, denom_base, device)

        # Transformer stack
        layer = tr.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff_dim)
        self.trf = tr.nn.TransformerEncoder(layer, nlayers)

        # Logit readout
        self.readout = tr.nn.Linear(d_model, len(vocab))

    def forward(self, seq):

        seq = self.embedder(seq)
        seq = seq.unsqueeze(1) # singleton batch dimension
        seq = self.positional_encoder(seq)
        seq = self.trf(seq) #, is_causal=True) # withhold future output
        return self.readout(seq).squeeze() # logits

def train_and_test():

    ### region print start
    #### for original format
    thms = load_database(15)
    for label, (assrt, essen, proof) in thms.items():
        print(f'{label} {assrt}')
        for (label, hypot) in essen:
            print(f'  {label} {hypot}')
        print(proof)
        print('-------------------------------------')

    #### for training format?
    thms = load_database(15)
    for prompt, output in get_samples(thms):
        print(" ".join(prompt)) ## prove tree path?
        print(" ".join(output)) ## ?
        break
    ### region print end

    """
    Load big chunk of the database
    """
    thms = load_database(10000)

    """
    data stats: max sequence length and vocab size
    """
    vocab = {"$=": 0}  # reuse this token as a delimiter between goals (input) and proof steps (output)
    seqlens = {}
    for s, (prompt, output) in enumerate(get_samples(thms)):

        # full sequence length
        seq = prompt + ["$="] + output

        # update vocabulary
        for tok in seq: vocab[tok] = vocab.get(tok, 0) + 1

        # update length distribution
        seqlens[len(seq)] = seqlens.get(len(seq), 0) + 1

        if s % 5000 == 0: print(f"{s} of {len(thms)}, |V| >= {len(vocab)}, L >= {max(seqlens.keys())}")

    # histogram of prompt lengths
    lengths = list(sorted(seqlens.keys()))
    counts = [seqlens[length] for length in lengths]

    pt.figure(figsize=(8, 3))

    pt.subplot(1, 2, 1)
    pt.plot(lengths, counts, 'k-')
    pt.xlabel("Prompt length")
    pt.ylabel("Frequency")

    # histogram of vocabulary
    pt.subplot(1, 2, 2)
    pt.plot(sorted(vocab.values()), 'k-')
    pt.yscale("log")
    pt.xlabel("Sorted token index")
    pt.ylabel("Frequency")

    pt.tight_layout()
    pt.show()


    embedder = Embedder(vocab, 64, device=device_laptop)  #tr.device("cuda:0"))
    print(embedder.get_index(["|-"]))
    print(embedder.get_tokens(embedder.get_index(["|-"])))
    print(embedder(["|-", "ph"]))

    PE_ex = make_positional_encodings(128, 256, denom_base=500)

    pt.figure(figsize=(4, 3))
    pt.subplot(1, 2, 1)
    pt.imshow(PE_ex)
    pt.ylabel('pos')
    pt.xlabel('sinusoid')

    pt.subplot(1, 2, 2)
    pt.imshow(PE_ex @ PE_ex.t())
    pt.title('Dot product similarity')
    pt.tight_layout()


    gpu = device_laptop # tr.device("cuda:0")
    model = Net(vocab, 128, 500, 64, 8, 2, 64, device=gpu).to(gpu)
    seq = prompt + ["$="] + output
    print(len(seq))
    logits = model(seq)
    print(len(seq))
    print(logits.shape)

    """
    Train/test split
    """
    examples = []
    for ex, (prompt, output) in enumerate(get_samples(thms)):
        if ex == 1000: break
        examples.append((prompt, output))

    random.shuffle(examples)
    train, test = examples[:900], examples[900:]

    """
    Training loop
    """

    #device = tr.device('cpu')
    #device = tr.device("cuda:0")

    max_len = 128
    d_model = 256
    nhead = 16
    nlayers = 6
    ff_dim = 4096
    denom_base = 500

    # noam schedule
    use_noam = True
    base_lr = 0.001
    warmup = 1000

    # init model (includes embedding and positional encoding)
    model = Net(vocab, max_len, denom_base, d_model, nhead, nlayers, ff_dim, device_laptop).to(device_laptop)

    # opt = tr.optim.SGD(model.parameters(), lr=base_lr)
    opt = tr.optim.Adam(model.parameters(), lr=base_lr)

    # vanilla cross-entropy
    loss_fn = tr.nn.CrossEntropyLoss()

    num_epochs = 20
    num_updates = 1000
    losses, accs = [], []
    test_accs = []
    total_updates = 0
    for epoch in range(num_epochs):

        # use random data order every epoch
        random.shuffle(train)

        # train on all the training data
        for update, (prompt, output) in enumerate(train):
            if update == num_updates: break

            # form training sequence, output shifted by one position
            seq = prompt + ["$="] + output
            inp, out = seq[:-1], seq[1:]

            # skip examples with conjectures longer than max
            if len(seq) >= max_len: continue

            opt.zero_grad()

            # send example through model
            logits = model(inp)  # predictions shifted by one
            targets = model.embedder.get_index(out)

            # grad descent on cross-entropy (only in output region)
            loss = loss_fn(logits[len(prompt):], targets[len(prompt):])
            loss.backward()
            opt.step()

            # Noam ramp-up
            if use_noam:
                upd = total_updates + 1
                lr = base_lr * min(upd ** (-0.5), upd * warmup ** (-1.5))
                for p in opt.param_groups: p['lr'] = lr

            # accuracy of one-step-ahead prediction (only in output region)
            preds = logits[len(prompt):].detach().argmax(dim=1)
            acc = (preds == targets[len(prompt):]).to(float).mean()

            # log results
            losses.append(loss.item())
            accs.append(acc.item())

            if update % 100 == 0:
                print('inp, targ, pred:')
                print(" ".join(inp[len(prompt):]))
                print(" ".join(out[len(prompt):]))
                print(" ".join(model.embedder.get_tokens(preds)))

                print(f"{epoch}|{update}: loss={losses[-1]}, acc={accs[-1]}")

            total_updates += 1

        # test on all the testing data
        test_acc = []
        for (prompt, output) in test:

            # form training sequence, output shifted by one position
            seq = prompt + ["$="] + output
            inp, out = seq[:-1], seq[1:]

            # skip examples with conjectures longer than max
            if len(seq) >= max_len: continue

            # send example through model
            with tr.no_grad():
                logits = model(inp)  # predictions shifted by one
                targets = model.embedder.get_index(out)

                # accuracy of one-step-ahead prediction (only in output region)
                preds = logits[len(prompt):].detach().argmax(dim=1)
                correct = (preds == targets[len(prompt):]).cpu().numpy()
                test_acc.append(correct.mean())

        test_accs.append(np.mean(test_acc))
        print(f"\n **** test acc = {test_accs[-1]} ****\n")

   # model, train, losses, accs, test_accs = None
    return model, train, losses, accs, test_accs

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
    return data_loader_basic.get_data_loader(2000, False)
    #return data_loader_basic.data_loader_clause()
    #return data_loader_basic.get_data_lemma()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DataLoader = get_data_loader()

    input("stucking for check ......")
    sys.exit(0)

    model, train, losses, accs, test_accs = train_and_test()
    show_and_the_rest(model, train, losses, accs, test_accs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
