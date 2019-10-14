import pickle
import numpy as np
import tensorflow as tf

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
def load_data(filename, regression=False, labels_filename=None):
    with open(filename, 'rb') as f:
        all_peps = np.load(f)
    stop_idx = (len(all_peps) * 4) // 5
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

def build_model(labels, hyperparam_pairs, regression):
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
                                                  activation=tf.nn.softmax if not regression else tf.math.sigmoid))
    # Instead of learner NN model, here we use uncertainty minimization
    # loss in the classifiers is number of misclassifications
    classifiers_losses = [tf.losses.absolute_difference(labels=labels_tensor,
                                                        predictions=x) for x in classifier_outputs]
    #[tf.losses.softmax_cross_entropy(onehot_labels=labels_tensor, logits=x) for x in classifier_outputs]
    total_classifiers_loss = tf.reduce_sum(classifiers_losses) / float(len(classifiers_losses))
    classifier_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_classifiers_loss)
    return [total_classifiers_loss, classifier_outputs, classifier_optimizer]
