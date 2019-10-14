import numpy as np
import pickle
from sys import argv
import tensorflow as tf
from utils import *

'''This script takes in a directory that should be full of vectorized peptides,
   which can be created from raw APD files via vectorize_peptides.py
   It pads all peptide sequences out to a max length of 200, which is taken
   from observing that the max length of a peptide in the APD dataset is 183.
   Padding recommended by TF devs:
   https://github.com/tensorflow/graphics/issues/2#issuecomment-497428806'''

def random_strategy(peps, est_labels, regression):
    return np.random.randint(0, len(peps))

def qbc_class_strategy(peps, est_labels):
    raw_variances = np.array([[item[0] * item[1] for item in predictions_arr] for predictions_arr in est_labels])
    variances = np.mean(raw_variances, axis=0)
    var_sum = np.sum(variances)
    probs_arr = variances/var_sum
    # now choose randomly, weighted by how much the models disagree
    # using len(peps) here ensures our probs arr is correct length
    chosen_idx = np.random.choice(range(len(peps)), p=probs_arr) 
    return chosen_idx

def qbc_regress_strategy(peps, est_labels):
    stdevs = np.std(np.array(est_labels), axis=0)
    probs_arr = stdevs[:,0] / np.sum(stdevs)
    # now choose randomly, weighted by how much the models disagree
    # using len(peps) here ensures our probs arr is correct length
    chosen_idx = np.random.choice(range(len(peps)), p=probs_arr) 
    return chosen_idx

def qbc_strategy(peps, est_labels, regression):
    if regression:
        return qbc_regress_strategy(peps, est_labels)
    return qbc_class_strategy(peps, est_labels)

def umin_strategy(peps, est_labels, regression):
    variances = [item[0] * item[1] for item in est_labels[0]]
    var_sum = np.sum(variances)
    chosen_idx = np.random.choice(range(len(peps)), p=[(item/var_sum) for item in variances])
    return chosen_idx

def printHelp():
    print('usage: active_learn.py '
          '[peptide_positive_vectors_file] '
          '[peptide_negative_vectors_file] '
          '[output_dirname] '
          '[strategy {all, random, qbc, umin}]'
          '[index]'
          '[regression: 0 or 1]'
          '(Use negative_vectors_file for activities file when doing regression.)')
    exit()


if __name__ == '__main__':
    if len(argv) < 6:
        printHelp()
        exit(1)

    positive_filename = argv[1] # positive example data is located here
    negative_filename = argv[2] # negative example data is located here
    output_dirname = argv[3] # where to save the output
    strategy_str = argv[4]
    index = argv[5] # which iteration are we on, will be prefixed to filenames.
    if len(argv) == 7:
        regression = bool(int(argv[6]))
    else:
        regression = False # default to not doing regression

    (labels, peps), (withheld_labels, withheld_peps) = prepare_data(positive_filename, negative_filename, regression)
    hyperparam_pairs = []

    if strategy_str == 'qbc':
        strategy = qbc_strategy
        for i in range(3,6):
            for j in range(3, 6):
                hyperparam_pairs.append((i, j))
    elif strategy_str == 'umin':
        strategy = umin_strategy
    elif strategy_str == 'random':
        strategy = random_strategy
    elif strategy_str == 'all':
        strategy = None
    else:
        print('Unknown strategy ', strategy_str)
        printHelp()
        exit(1)
    hyperparam_pairs.append((5,6))

    learner = Learner(labels.shape[1], hyperparam_pairs, regression)
    evaluate_strategy((labels, peps), (withheld_labels, withheld_peps), learner,
                   output_dirname, strategy=strategy, index=index, regression=regression)