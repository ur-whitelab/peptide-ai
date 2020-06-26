import numpy as np
import pickle
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

DEFAULT_CONVOLUTION_HYPERPARAMS = (6,6)# found to give best performance overall

def random_strategy(peps, est_labels, regression):
    return np.random.randint(0, len(peps))

def qbc_class_strategy(peps, est_labels, stochastic=True):
    raw_variances = np.array([[item[0] * (1.0 - item[0]) for item in predictions_arr] for predictions_arr in est_labels])
    variances = np.mean(raw_variances, axis=0)
    var_sum = np.sum(variances)
    probs_arr = variances/var_sum
    # now choose randomly, weighted by how much the models disagree
    # using len(peps) here ensures our probs arr is correct length
    if stochastic:
        p_arr = variances/var_sum if var_sum > 0. else np.ones_like(variances)/len(variances)
        chosen_idx = np.random.choice(range(len(peps)), p=p_arr)
    else:
        chosen_idx = np.argmax(probs_arr)
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

def umin_strategy(peps, est_labels, regression, stochastic=True):
    print('IN UMIN_STRATEGY, est_labels is {}'.format(est_labels))
    variances = est_labels[0] * (np.ones_like(est_labels[0]) - est_labels[0])#[item[0] * (1. - item[0]) for item in est_labels[0]]
    print('IN UMIN_STRATEGY, variances is {}'.format(variances))
    var_sum = np.sum(variances)
    print('IN UMIN_STRATEGY, var_sum is {}'.format(var_sum))
    if stochastic:
        p_arr = variances/var_sum if var_sum > 0.0 else np.ones_like(variances)/len(variances)
        print('p_arr is {} ({})'.format(p_arr, p_arr.shape))
        chosen_idx = np.random.choice(range(len(peps)), p=p_arr.flatten())
    else:
        chosen_idx = np.argmax(variances)
    return chosen_idx

def get_active_learner(strategy_str, stochastic=True, convolution_params=None):
    hyperparam_pairs = []
    if strategy_str == 'qbc':
        strategy = lambda p, l, r: qbc_strategy(p, l, r, stochastic)
        strategy = qbc_strategy
        for i in range(3,6):
            for j in range(3, 6):
                hyperparam_pairs.append((i, j))
    elif strategy_str == 'umin':
        strategy = lambda p, l, r: umin_strategy(p, l, r, stochastic)
    elif strategy_str == 'random':
        strategy = random_strategy
    elif strategy_str == 'all':
        strategy = None
    else:
        print('Unknown strategy ', strategy_str)
        printHelp()
        exit(1)
    if convolution_params is None:
        hyperparam_pairs.append(DEFAULT_CONVOLUTION_HYPERPARAMS)
    else:
        hyperparam_pairs.append(convolution_params)
    return strategy, hyperparam_pairs

def printHelp():
    print('usage: active_learn.py '
          '[data_root] '
          '[output_dirname] '
          '[N_samples] '
          '[dataset_index] '
          '[strategy {all, random, qbc, umin}]'
          '[regression: 0 or 1]'
          '(Use negative_vectors_file for activities file when doing regression.)')
    exit()


if __name__ == '__main__':
    if len(argv) < 6:
        printHelp()
        exit(1)

    root = argv[1] # location of data
    output_dirname = argv[2] # where to save the output
    NSAMPLES = int(argv[3]) # how many samples (individual datapoints)
    dataset_choice = argv[4] # index of chosen dataset
    strategy_str = argv[5]
    regression = False # default to not doing regression
    if len(argv) > 6:
        regression = bool(int(argv[6]))
        if len(argv) > 7:
            convolution_params = (int(argv[7]), int(argv[8]))
    else:
        regression = False
        convolution_params = None
    datasets = load_datasets(root)
    name, (labels, peps), (withheld_labels, withheld_peps) = datasets[int(dataset_choice)]

    strategy, hyperparam_pairs = get_active_learner(strategy_str, convolution_params=convolution_params)
    learner = Learner(1, hyperparam_pairs, regression)

    odir = os.path.join(output_dirname + '-' + str(NSAMPLES), strategy_str, dataset_choice)
    os.makedirs(odir, exist_ok=True)
    nruns = NSAMPLES
    ntrajs = 100
    batch_size = 16
    if strategy is None:
        nruns = 1000000 # just go big
        ntrajs = 10
        batch_size = 32
    for i in range(ntrajs):
        # re-split data
        (labels, peps), (withheld_labels, withheld_peps) = mix_split([peps, withheld_peps], [labels, withheld_labels])
        evaluate_strategy((labels, peps), (withheld_labels, withheld_peps), learner,
                   odir, strategy=strategy, nruns=nruns, index=i, regression=regression, batch_size=batch_size)
