import numpy as np
import math
from pep_generator import pep_seq
from IPython.display import clear_output

# parameter setting
T0 = 10
dT = -0.005

def score(seq): # num of valine
    num = 0
    for x in seq:
        if x == 'V':
            num += 1       
    return 100*num/len(seq)

def factorial(n):
    a = 1
    for i in range(1,n+1):
        a = a*i
    return a

# the sequence length follow poisson distribution
def pre_ex(seq1,seq2): # not used for now
    mu = 10
    k1 = len(seq1)
    k2 = len(seq2)
    p = mu**(k1-k2) / (factorial(k1)/factorial(k2))
    return p

# Simulated Annealing

pep = pep_seq(ALPHABET, 5, 100) # length range (5,100)
pep.init(15) # init the pep length to 15
T = T0
for _ in range(int(-T0/dT)-10):
    T += dT
    pep_tmp = np.copy(pep.nxt(np.random.randint(0,3)))
    if len(pep_tmp)<5 or len(pep_tmp)>100:
        continue
    d_score = (-score(pep_tmp)) - (-score(pep.seq))
    if d_score < 0 or math.exp(-d_score/T) > np.random.rand() :
        pep.seq = np.copy(pep_tmp)
        print(pep.seq)
        x = system('clear')
        if _ != int(-T0/dT)-10:
            clear_output(wait=True)