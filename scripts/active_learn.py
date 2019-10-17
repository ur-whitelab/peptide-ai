import numpy as np
import pickle
import tqdm
from sys import argv
import tensorflow as tf
from utils import *
import os

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

def get_active_learner(strategy_str):
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
    return strategy, hyperparam_pairs

def printHelp():
    print('usage: active_learn.py '
          '[data_root] '
          '[output_dirname] '
          '[dataset_index] '
          '[strategy {all, random, qbc, umin}]'
          '[regression: 0 or 1]'
          '(Use negative_vectors_file for activities file when doing regression.)')
    exit()


if __name__ == '__main__':
    if len(argv) < 5:
        printHelp()
        exit(1)

    root = argv[1] # location of data
    output_dirname = argv[2] # where to save the output
    dataset_choice = argv[3] # index of chosen dataset
    strategy_str = argv[4]
    regression = False # default to not doing regression
    if len(argv) > 5:
        regression = bool(int(argv[5]))
    else:
        regression = False
    datasets = load_datasets(root)
    name, (labels, peps), (withheld_labels, withheld_peps) = datasets[int(dataset_choice)]

    strategy, hyperparam_pairs = get_active_learner(strategy_str)
    learner = Learner(2, hyperparam_pairs, regression)

    odir = os.path.join(output_dirname, strategy_str, dataset_choice)
    os.makedirs(odir, exist_ok=True)
    nruns = 10
    ntrajs = 250
    if strategy is None:
        nruns = 10000 # just go big
        ntrajs = 1
    for i in tqdm.tqdm(range(ntrajs)):
        # randomly swap labels
        if np.random.uniform() < 0.5:
            labels[:,0] = 1 - labels[:,0]
            labels[:,1] = 1 - labels[:,1]
            withheld_labels[:,0] = 1 - withheld_labels[:,0]
            withheld_labels[:,1] = 1 - withheld_labels[:,1]
        evaluate_strategy((labels, peps), (withheld_labels, withheld_peps), learner,
                   odir, strategy=strategy, nruns=nruns, index=i, regression=regression)
