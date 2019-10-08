import numpy as np
import pickle
from sys import argv
import tensorflow as tf
from vectorize_peptides import vectorize
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
from utils import *

'''This script takes in a directory that should be full of vectorized peptides,
   which can be created from raw APD files via vectorize_peptides.py
   It pads all peptide sequences out to a max length of 200, which is taken
   from observing that the max length of a peptide in the APD dataset is 183.
   Padding recommended by TF devs:
   https://github.com/tensorflow/graphics/issues/2#issuecomment-497428806'''

NRUNS = 5000 # number of training runs to do

BATCH_SIZE = 20 # number of peptides to look at for one training iteration

def printHelp():
    print('usage: test_maml_convolution.py '
          '[peptide_positive_vectors_file (str)] '
          '[peptide_negative_vectors_file (str)] '
          '[output_dirname (str)] '
          '[index (int)] '
          '[regression: (0 or 1)] '
          '(Use negative_vectors_file for activities file when doing regression.)')
    exit()

if len(argv) != 6:
    printHelp()
    exit(1)

positive_filename = argv[1] # positive example data is located here
negative_filename = argv[2] # negative example data is located here
output_dirname = argv[3] # where to save the output
INDEX = argv[4] # which iteration are we on, will be prefixed to filenames.
regression = bool(int(argv[5]))

# load randomly shuffled training data
if regression:
    positive_peps, positive_withheld_peps, labels, withheld_labels = load_data(positive_filename, regression, negative_filename)
# negative dataset only needed if we're classifying
else:
    positive_peps, positive_withheld_peps = load_data(positive_filename, regression)
    negative_peps, negative_withheld_peps = load_data(negative_filename, regression)

peps = np.concatenate([positive_peps, negative_peps]) if not regression else positive_peps
withheld_peps = np.concatenate([positive_withheld_peps, negative_withheld_peps]) if not regression else positive_withheld_peps

# calculate activities if we're not doing regression
if not regression:
    labels = np.zeros([len(peps), 2]) # one-hot encoding of labels
    labels[:len(positive_peps),0] += 1. # positive labels are 0 index
    labels[len(positive_peps):,1] += 1. # negative labels are 1 index
    withheld_labels = np.zeros([len(withheld_peps), 2]) # one-hot encoding of labels
    withheld_labels[:len(positive_withheld_peps),0] += 1. # positive labels are 0 index
    withheld_labels[len(positive_withheld_peps):,1] += 1. # negative labels are 1 index

# now re-shuffle all these in the same way
shuffle_same_way([labels, peps])
shuffle_same_way([withheld_labels, withheld_peps])

#prepare 10 sets of hyperparameter pairs for the task model
hyperparam_pairs = []
#for i in range(3,6):
#    for j in range(3, 6):
#        hyperparam_pairs.append((i, j))
hyperparam_pairs.append((5,6))

train_losses = []
withheld_losses = []
pep_choice_indices = []

final_withheld_predictions = None
with tf.Session() as sess:
    total_classifiers_loss, classifier_outputs, classifier_optimizer = build_model(labels, hyperparam_pairs, regression)
    # here is where the sessions are set up and called
    sess.run(tf.global_variables_initializer())
    # get the initial losses
    train_loss = sess.run([total_classifiers_loss],
                          feed_dict={
                              'input:0': peps,
                              'labels:0': labels
                          })
    train_losses.append(train_loss[0])
    withheld_loss = sess.run([total_classifiers_loss],
                             feed_dict={
                                 'input:0': withheld_peps,
                                 'labels:0': withheld_labels,
                                 'dropout_rate:0': 0.0
                             })
    withheld_losses.append(withheld_loss)
    for i in tqdm(range(NRUNS)): # essentially a batch size of 1
        # train the classifier with all peptides
        random_indices = []
        for j in range(BATCH_SIZE):
            random_idx = np.random.randint(0, len(peps))
            random_indices.append(random_idx)
            pep_choice_indices.append(random_idx)
        random_indices = set(random_indices)
        opt_output = sess.run([classifier_optimizer, total_classifiers_loss],
                              feed_dict={
                                  'input:0': [peps[idx] for idx in random_indices],
                                  'labels:0': [labels[idx] for idx in random_indices]
                              })
        # get the loss
        train_loss = opt_output[1]
        train_losses.append(train_loss)
        withheld_loss = sess.run([total_classifiers_loss],
                              feed_dict={
                                  'input:0': withheld_peps,
                                  'labels:0': withheld_labels,
                                  'dropout_rate:0': 0.0
                              })
        withheld_losses.append(withheld_loss)
    print('RUN FINISHED. CHECKING LOSS ON WITHHELD DATA.')
    # now that training is done, get final withheld predictions
    final_withheld_predictions = sess.run(classifier_outputs,
                                          feed_dict={
                                              'input:0': withheld_peps,
                                              'labels:0': withheld_labels,
                                              'dropout_rate:0': 0.0
                                          })

np.savetxt('{}/{}_train_losses.txt'.format(output_dirname, INDEX.zfill(4)), train_losses)
np.savetxt('{}/{}_withheld_losses.txt'.format(output_dirname, INDEX.zfill(4)), withheld_losses)
np.savetxt('{}/{}_choices.txt'.format(output_dirname, INDEX.zfill(4)), pep_choice_indices)

# can't do ROC for regression
if not regression:
    # AUC analysis (misclassification) for final withheld predictions
    withheld_aucs = []
    withheld_fprs = []
    withheld_tprs = []
    withheld_thresholds = []
    # iterate over all models
    for i, predictions_arr in enumerate(final_withheld_predictions):
        withheld_fpr, withheld_tpr, withheld_threshold = roc_curve(withheld_labels[:,0],
                                                                    predictions_arr[:,0])
        withheld_fprs.append(withheld_fpr)
        withheld_tprs.append(withheld_tpr)
        withheld_thresholds.append(withheld_threshold)
        withheld_auc = auc(withheld_fpr, withheld_tpr)
        withheld_aucs.append(withheld_auc)
        np.save('{}/{}_fpr_{}.npy'.format(output_dirname, INDEX.zfill(4), i), withheld_fpr)
        np.save('{}/{}_tpr_{}.npy'.format(output_dirname, INDEX.zfill(4), i), withheld_tpr)
        np.save('{}/{}_thresholds_{}.npy'.format(output_dirname, INDEX.zfill(4), i), withheld_thresholds)
    np.savetxt('{}/{}_auc.txt'.format(output_dirname, INDEX.zfill(4)), withheld_aucs)
