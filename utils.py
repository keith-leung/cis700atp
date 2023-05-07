import matplotlib.pyplot as pt
import numpy as np
import torch as tr
import random


set_mm_path = 'C:/Users/Public/Documents/metamath/set.mm'
in_memory_cache_thms = {}

"""
Load database into memory
thms[label] = (assertion, essential hypotheses, proof inferences)
"""
def load_database(max_thms = None):

    with open(set_mm_path, "r") as f: raw = f.read()

    # collect theorems
    thms = {}

    # parse
    tags = "cfeap="
    mark = {}
    scope = [[]]
    comment = False
    last = ""
    for i in range(2, len(raw)):
        '''
        omit all comments, only extract useful elements
        '''
        # comments
        if raw[i-2:i] == "$(": comment = True
        if raw[i-2:i] == "$)": comment = False
        if comment: continue

        '''
        determine what cascade scope the current level is,
        use a "stack" scope to determine by []
        '''
        # scope
        if raw[i-2:i] == "${": scope.append([])
        if raw[i-2:i] == "$}": scope.pop()


        '''
        only cfeap= are useful elements: $c , $f , $e , $a , $p , $=
        '''
        # mark simple tags
        for t in tags:
            if raw[i-2:i] == "$"+t:
                last = t
                mark[t] = i

        # extract
        if raw[i-2:i] == "$.":
            # essential
            if last == 'e':
                hypot = raw[mark['e']+1:i-2]
                label = raw[raw[:mark['e']].rindex('\n'):mark['e']-2].strip()
                scope[-1].append((label, hypot))
            # axiom
            if last == 'a':
                assrt = raw[mark['a']+1:i-2]
                label = raw[raw[:mark['a']].rindex('\n'):mark['a']-2].strip()
                essen = tuple(e for s in scope for e in s)
                thms[label] = (assrt, essen, ())
                if label == 'stoic4b': break

            # theorem
            if last == '=':
                # proof = tuple(raw[mark['=']+1:i-3].split()) # uncompressed
                proof = raw[mark['=']+1:i-3]
                proof = tuple(proof[proof.find("(")+2:proof.find(")")-1].split())
                assrt = raw[mark['p']+1:mark['=']-2]
                label = raw[raw[:mark['p']].rindex('\n'):mark['p']-2].strip()
                essen = tuple(e for s in scope for e in s)
                thms[label] = (assrt, essen, proof)
                if label == 'stoic4b': break

            if last in 'a=' and len(thms) % 1000 == 0:
                print(f"{len(thms)} thms parsed...")
            if len(thms) == max_thms: break

    return thms


def tokenize(assrt, essen):
    tokens = assrt.split()
    for (label, hypot) in essen:
        tokens.extend(hypot.split())
    return tokens

def get_samples(thms):
    for label, (assrt, essen, proof) in thms.items():

        # skip axioms, they do not have proofs
        if len(proof) == 0: continue

        # lookup assertion of last proof step
        proof_assrt, proof_essen, _ = thms[proof[-1]]

        # tokenize input/output
        prompt = tokenize(assrt, essen)
        output = tokenize(proof_assrt, proof_essen)

        yield prompt, output


"""
sinusoidal position encoding module
"""
def make_positional_encodings(max_len, d_model, denom_base):
    d2 = d_model // 2
    PE = tr.cat([
        trig(tr.arange(max_len).unsqueeze(1) / denom_base**(tr.arange(d2)/d2))
        for trig in (tr.sin, tr.cos)], dim=1)
    return PE

def trigonometric_positional_encoder(max_len, d_model, denom_base, device):
    PE = make_positional_encodings(max_len, d_model, denom_base)
    PE = PE.unsqueeze(1) # account for batch dimension
    PE = PE.to(device)
    return lambda inputs: inputs + PE[:len(inputs)]
