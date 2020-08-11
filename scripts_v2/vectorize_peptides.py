import numpy as np
import pickle
from sys import argv

'''This script is used to transform raw APD output into one-hot vectors 
   for machine learning. Input: raw APD output file and location to save 
   vectorized peptides as .npy files. Output: an atlas file with APD names
   and corresponding vector filenames, and .npy files for all vectorized peptides'''

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

def vectorize(pep):
    '''Takes a string of amino acids and encodes it to an L x 20 one-hot vector,
       where L is the length of the peptide.'''
    vec = np.zeros((len(pep), 20))
    for i, letter in enumerate(pep):
        vec[i][ALPHABET.index(letter)] += 1.
    return(vec)

def main():
    if len(argv) != 3:
        print('usage: vectorize_peptides.py [raw_apd_text_file] [save_directory]')
        exit()
    
    fname = argv[1]
    output_dir = argv[2]
    
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
    
    names = []
    sequences = []
    filenames = []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            names.append(line)
            filenames.append(line+'.npy')
        else:
            sequences.append(line.split()[-1])
    
    with open('{}/atlas.pb'.format(output_dir), 'wb') as f:
        pickle.dump(list(zip(names, sequences)), f)
    
    #now create one-hot encodings for each sequence.
    for seq, fname in zip(sequences, filenames):
        np.save('{}/{}'.format(output_dir, fname), vectorize(seq))

if __name__ == '__main__':
    main()
