import pickle
import numpy as np
import tensorflow as tf
from vectorize_peptides import vectorize
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

MAX_LENGTH = 200
ALPHABET_SIZE = len(ALPHABET)
HIDDEN_LAYER_SIZE = 64
DEFAULT_DROPOUT_RATE = 0.0
LEARNING_RATE = 0.001

def shuffle_same_way(list_of_arrays):
    rng_state = np.random.get_state()
    for arr in list_of_arrays:
        np.random.shuffle(arr)
        np.random.set_state(rng_state)

# expect to be pointed to a single big .npy file with all the
# one-hot encoded peptides in it
def load_data(filename, regression=False, labels_filename=None, withheld_percent=0.20):
    with open(filename, 'rb') as f:
        all_peps = np.load(f)
    stop_idx = int(len(all_peps) * (1 - withheld_percent))
    print('Loading {} peptides from {} (witholding {})'.format(stop_idx, filename, len(all_peps) - stop_idx))
    if regression and labels_filename is not None:
        all_labels = np.load(open(labels_filename, 'rb')).astype(np.float64)
        all_peps, all_labels = all_peps[np.abs(all_labels - np.mean(all_labels)) < 2. * np.std(all_labels)], all_labels[np.abs(all_labels - np.mean(all_labels)) < 2. * np.std(all_labels)]
        all_labels = -np.log(all_labels)
        all_labels -= np.min(all_labels) # normalize to have a min of 0.0
        all_labels /= np.max(all_labels) # normalize to have max of 1.0
        print('labels is now: {}'.format(all_labels))
        np.savetxt('all_labels.txt', all_labels)
        shuffle_same_way([all_peps, all_labels])
    else:
        np.random.shuffle(all_peps)
    peps = all_peps[:stop_idx] #np.zeros([len(all_peps[:stop_idx]), MAX_LENGTH, ALPHABET_SIZE])
    withheld_peps = all_peps[stop_idx:] #np.zeros([len(all_peps[stop_idx:]), MAX_LENGTH, ALPHABET_SIZE])
    retval = [peps, withheld_peps]
    if regression:
        labels, withheld_labels = all_labels[:len(peps)], all_labels[len(peps):]
        retval += [np.reshape(labels, [len(labels), 1]),
                   np.reshape(withheld_labels, [len(withheld_labels), 1])]
    return retval

def make_convolution(motif_width, num_classes, input_tensor):
    data_tensor = input_tensor
    filter_tensor = tf.Variable(np.random.random([motif_width,# filter width
                                                  ALPHABET_SIZE,# in channels
                                                  num_classes]),# out channels 
                                name='filter_{}_width_{}_classes'.format(
                                    motif_width,
                                    num_classes),
                                dtype=tf.float64)
    conv = tf.nn.conv1d(data_tensor,
                        filter_tensor,
                        padding='SAME', # keep the dimensions of output the same
                        stride=1, # look at each residue
                        name='convolution_{}_width_{}_classes'.format(
                            motif_width,
                            num_classes))
    output = tf.math.reduce_max(conv, axis=1)
    return output

def build_model(label_width, hyperparam_pairs, regression):
    input_tensor = tf.placeholder(shape=np.array((None, MAX_LENGTH, ALPHABET_SIZE)),
                                  dtype=tf.float64,
                                  name='input')
    labels_tensor = tf.placeholder(shape=(input_tensor.shape[0], label_width),
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
                                                  label_width,
                                                  activation=tf.nn.softmax if not regression else tf.math.sigmoid))
    # Instead of learner NN model, here we use uncertainty minimization
    # loss in the classifiers is number of misclassifications
    classifiers_losses = [tf.losses.absolute_difference(labels=labels_tensor,
                                                        predictions=x) for x in classifier_outputs]
    #[tf.losses.softmax_cross_entropy(onehot_labels=labels_tensor, logits=x) for x in classifier_outputs]
    total_classifiers_loss = tf.reduce_sum(classifiers_losses) / float(len(classifiers_losses))
    classifier_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_classifiers_loss)
    return [total_classifiers_loss, classifier_outputs, classifier_optimizer]

class Learner:
    def __init__(self, label_width, hyperparam_pairs, regression=False):
        model_vars = build_model(
                label_width, 
                hyperparam_pairs, 
                regression)
        self.total_classifiers_loss = model_vars[0]
        self.classifier_outputs = model_vars[1]
        self.classifier_optimizer = model_vars[2]
        
    def train(self, sess, labels, peps, iters=10, batch_size=5, replacement=True):
        losses = [0 for _ in range(iters)]
        for i in range(iters):
            indices = np.random.choice(peps.shape[0], batch_size, replace=replacement)
            losses[i], _ = sess.run(
                [self.total_classifiers_loss, self.classifier_optimizer],
                feed_dict={'input:0': peps[indices], 'labels:0': labels[indices]})
        return losses
    
    def eval_loss(self, sess, labels, peps):
        return self.eval(sess, labels, peps)[0]

    def eval_labels(self, sess, peps):
        return sess.run(self.classifier_outputs, 
            feed_dict={'input:0': peps, 'dropout_rate:0': 0.0})

    def eval(self, sess, labels, peps):
        return sess.run(
            [self.total_classifiers_loss, self.classifier_outputs],
            feed_dict={'input:0': peps, 'labels:0': labels, 'dropout_rate:0': 0.0})


def prepare_data(positive_filename, negative_filename, regression, withheld_percent=0.2):
    # load randomly shuffled training data
    if regression:
        positive_peps, positive_withheld_peps, labels, withheld_labels = load_data(positive_filename, regression, negative_filename, withheld_percent=withheld_percent)
    # negative dataset only needed if we're classifying
    else:
        positive_peps, positive_withheld_peps = load_data(positive_filename, regression, withheld_percent=withheld_percent)
        negative_peps, negative_withheld_peps = load_data(negative_filename, regression, withheld_percent=withheld_percent)

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

    return (labels, peps), (withheld_labels, withheld_peps)



def evaluate_strategy(train_data, withheld_data, learner, output_dirname, strategy=None,
                      nruns=10, index=0, regression=False):
    peps = train_data[1]
    labels = train_data[0]
    withheld_peps = withheld_data[1]
    withheld_labels = withheld_data[0]

    train_losses = []
    withheld_losses = []
    pep_choice_indices = []


    with tf.Session() as sess:
        # here is where the sessions are set up and called
        sess.run(tf.global_variables_initializer())
        # do not get initial loss, just skip into algorithm
        for i in tqdm(range(nruns)):
            # run the classifiers on all available peptides to get their outputs
            # then pick the one with the highest variance (p*(1-p)) to train with
            output = learner.eval_labels(sess, peps)
            # make random selections for next training point.
            if strategy is None:
                train_losses.append(learner.train(sess, labels, peps)[-1])
            else:
                chosen_idx = strategy(peps, output, regression)
                pep_choice_indices.append(chosen_idx)
                # train for the chosen number of steps after each observation
                # only append final training value
                train_losses.append(learner.train(sess, labels[pep_choice_indices], peps[pep_choice_indices])[-1])
            withheld_losses.append(learner.eval_loss(sess, withheld_labels, withheld_peps))
                
        print('RUN FINISHED. CHECKING LOSS ON WITHHELD DATA.')
        # now that training is done, get final withheld predictions
        final_withheld_predictions = learner.eval_labels(sess, withheld_peps)
        final_train_predictions = learner.eval_labels(sess, peps)

    np.savetxt('{}/{}_train_losses.txt'.format(output_dirname, index.zfill(4)), train_losses)
    np.savetxt('{}/{}_withheld_losses.txt'.format(output_dirname, index.zfill(4)), withheld_losses)
    np.savetxt('{}/{}_choices.txt'.format(output_dirname, index.zfill(4)), pep_choice_indices)

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
            np.save('{}/{}_fpr_{}.npy'.format(output_dirname, index.zfill(4), i), withheld_fpr)
            np.save('{}/{}_tpr_{}.npy'.format(output_dirname, index.zfill(4), i), withheld_tpr)
            np.save('{}/{}_thresholds_{}.npy'.format(output_dirname, index.zfill(4), i), withheld_thresholds)
        np.savetxt('{}/{}_auc.txt'.format(output_dirname, index.zfill(4)), withheld_aucs)
    # for regression, instead rank the training set and record results.
    else:
        final_predictions = []
        for prediction in final_withheld_predictions:
            final_predictions.append(prediction)
        for prediction in final_train_predictions:
            final_predictions.append(prediction)
        output_peps = [item for item in withheld_peps]
        for item in peps:
            output_peps.append(item)
        np.save('{}/{}_final_peps.npy'.format(output_dirname, index.zfill(4)), output_peps)
        np.save('{}/{}_final_predictions.npy'.format(output_dirname, index.zfill(4)), final_predictions)