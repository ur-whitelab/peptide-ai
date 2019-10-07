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


def printHelp():
    print('usage: active_learner_uncertainty_min.py '
          '[peptide_positive_vectors_file (str)] '
          '[peptide_negative_vectors_file (str)] '
          '[output_dirname (str)] '
          '[index (int)]'
          '[regression: (0 or 1)] '
          '(Use negative_vectors_file for activities file when doing regression.)')
    exit()

if len(argv) != 6:
    printHelp()
    exit(1)

positive_filename = argv[1] # positive example data is located here
negative_filename= argv[2] # negative example data is located here
output_dirname = argv[3] # where to save the output
INDEX = argv[4] # which iteration are we on, will be prefixed to filenames.
regression = bool(int(argv[5]))
TRAIN_ITERS_PER_SAMPLE = 50

LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.0
NRUNS = 50
HIDDEN_LAYER_SIZE = 64

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
    # peptide sequences input
    # shape is [N_data_points, MAX_LENGTH, ALPHABET_SIZE], flexible in N_data_points
    input_tensor = tf.placeholder(shape=np.array((None, MAX_LENGTH, ALPHABET_SIZE)),
                                  dtype=tf.float64,
                                  name='input')
    labels_tensor = tf.placeholder(shape=(input_tensor.shape[0], labels.shape[1]),
                                   dtype=tf.int32,
                                   name='labels')
    dropout_rate = tf.placeholder_with_default(tf.constant(DEFAULT_DROPOUT_RATE, dtype=tf.float64), shape=(), name='dropout_rate')
    aa_counts = tf.reduce_sum(input_tensor, axis=1) # just add up the one-hots
    convs = []
    for hparam_pair in hyperparam_pairs:
        convs.append(make_convolution(hparam_pair[0], hparam_pair[1], input_tensor))
    classifier_conv_inputs = []
    classifier_inputs = []
    classifier_hidden_layers = []
    classifier_outputs = []
    for i, conv in enumerate(convs):
        classifier_conv_inputs.append(
            tf.nn.dropout(tf.layers.dense(conv,
                                          HIDDEN_LAYER_SIZE,
                                          activation=tf.nn.relu),
                          rate=dropout_rate)
        )
        classifier_inputs.append(tf.concat([classifier_conv_inputs[i],
                                            aa_counts],
                                           1)
        )
        classifier_hidden_layers.append(tf.nn.dropout(tf.layers.dense(classifier_inputs[i],
                                                                      HIDDEN_LAYER_SIZE,
                                                                      activation=tf.nn.tanh),
                                                      rate=dropout_rate)
        )
        # output is 2D: probabilities of 'has this property'/'does not have'
        # easier to compare logits with one-hot labels this way
        classifier_outputs.append(tf.layers.dense(classifier_hidden_layers[i],
                                                  labels.shape[1],
                                                  activation=tf.nn.softmax))
    # Instead of learner NN model, here we use uncertainty minimization
    # loss in the classifiers is number of misclassifications
    classifiers_losses = [tf.losses.absolute_difference(labels=labels_tensor, predictions=x) for x in classifier_outputs]
    #[tf.losses.softmax_cross_entropy(onehot_labels=labels_tensor, logits=x) for x in classifier_outputs]
    total_classifiers_loss = tf.reduce_sum(classifiers_losses)
    classifier_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_classifiers_loss)

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
    for i in tqdm(range(NRUNS)):
        # run the classifiers on all available peptides to get their outputs
        # then pick the one with the highest variance (p*(1-p)) to train with, then
        # train for TRAIN_ITERS_PER_SAMPLE iterations, sampling from past observations
        output = sess.run(classifier_outputs + \
                          [total_classifiers_loss],
                          feed_dict={
                              'input:0': peps,
                              'labels:0': labels
                          })
        # get which peptide has the highest variance.
        classifier_predictions = output[0]# special case where we only have one "consensus model"
        # TODO: fix this for regression!
        variances = [item[0] * item[1] for item in classifier_predictions]
        var_sum = np.sum(variances)
        chosen_idx = np.random.choice(range(len(variances)), p=[(item/var_sum) for item in variances])
        pep_choice_indices.append(chosen_idx)
        # train for the chosen number of steps after each observation
        for j in range(TRAIN_ITERS_PER_SAMPLE):
            # train the classifier, sampling from previously seen peptide(s)
            opt_output = sess.run([classifier_optimizer, total_classifiers_loss],
                                     feed_dict={
                                         'input:0': [peps[chosen_idx]],
                                         'labels:0': [labels[chosen_idx]]
                                     })
            
            # pick a random peptide to train on for next step
            chosen_idx = np.random.choice(pep_choice_indices)
        train_losses.append(opt_output[1])
        withheld_loss = sess.run([total_classifiers_loss],
                                 feed_dict={
                                     'input:0': withheld_peps,
                                     'labels:0': withheld_labels,
                                     'dropout_rate:0': 0.0
                                 })
        withheld_losses.append(withheld_loss)
    print('RUN FINISHED. CHECKING LOSS ON WITHHELD DATA.')
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
        np.save('{}/{}_fpr_{}.txt'.format(output_dirname, INDEX.zfill(4), i), withheld_fpr)
        np.save('{}/{}_tpr_{}.txt'.format(output_dirname, INDEX.zfill(4), i), withheld_tpr)
        np.save('{}/{}_thresholds_{}.txt'.format(output_dirname, INDEX.zfill(4), i), withheld_thresholds)
    np.savetxt('{}/{}_auc.txt'.format(output_dirname, INDEX.zfill(4)), withheld_aucs)
