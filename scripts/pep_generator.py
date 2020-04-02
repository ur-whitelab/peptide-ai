import numpy as np
import warnings
import math

class pep_seq():
    def __init__(self, ALPHABET, short, long):
        self.ALPHABET = ALPHABET
        self.len_range = range(short, long+1)
        self.seq = []
    
    def init(self, init_length):
        if init_length not in self.len_range:
            return warnings.warn("input length out of range", UserWarning)
        #init_length = np.random.choice(self.len_range)
        self.seq = list(np.random.choice(self.ALPHABET, init_length))
        return self.seq
        
    def add(self): #randomly pick a position and insert an amino acid
        seq = list(np.copy(self.seq))
        seq.insert(np.random.choice(len(seq)), np.random.choice(self.ALPHABET))
        return seq
        
    def remove(self): #randomly pick a position and remove an amino acid
        seq = list(np.copy(self.seq))
        seq.pop(np.random.choice(len(seq)))
        return seq
    
    def replace(self): #randomly pick a position and replace an amino acid with an other ramdom amino acid
        seq = list(np.copy(self.seq))
        seq[np.random.choice(len(seq))] = np.random.choice(self.ALPHABET)
        return seq
    
    def nxt(self, operation): #operation = 0(add), 1(remove) or 2(replace)
        if operation == 0: 
            result = self.add()
        if operation == 1: 
            result = self.remove()
        if operation == 2:
            result = self.replace()
        return result