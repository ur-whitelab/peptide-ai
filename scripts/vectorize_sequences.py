import sys
from vectorize_peptides import ALPHABET
import numpy as np

def encode(seq, max_length=200):
    data = np.zeros((max_length, 20), dtype=np.float)
    for i,s in enumerate(seq):
        data[i, ALPHABET.index(s)] = 1
    return data

with open(sys.argv[1], 'r') as f:
    vectors = [encode(s[:-1]) for s in f.readlines()]

data = np.concatenate([v[np.newaxis, :, :] for v in vectors], axis=0)
print(data.shape)
np.save(sys.argv[2], data, allow_pickle=False)