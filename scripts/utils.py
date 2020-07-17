import os
import pickle
import numpy as np
import tensorflow as tf
import random
from vectorize_peptides import vectorize
from sklearn.metrics import auc, roc_curve

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']

MAX_LENGTH = 200
ALPHABET_SIZE = len(ALPHABET)
HIDDEN_LAYER_SIZE = 64
HIDDEN_LAYER_NUMBER = 3
DEFAULT_DROPOUT_RATE = 0.0
MODEL_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_ITERS = 16

def shuffle_same_way(list_of_arrays):
    rng_state = np.random.get_state()
    for arr in list_of_arrays:
        np.random.shuffle(arr)
        np.random.set_state(rng_state)

def vec_to_seq(pep_vector):
    seq = ''
    # expect a 2D numpy array (pep_length x 20), give the string it represents
    for letter in pep_vector[:int(np.sum(pep_vector))]:
        idx = np.argmax(letter)
        if letter[idx] == 0:
            break
        seq += ALPHABET[idx]
    return(seq)

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

class Learner:
    def __init__(self, label_width, hyperparam_pairs, regression=False, learning_rate=MODEL_LEARNING_RATE):
        tf.compat.v1.disable_eager_execution()
        self.filter_tensors = []
        model_vars = self.build_model(
                label_width,
                hyperparam_pairs,
                regression,
                learning_rate)

    def make_convolution(self,motif_width, num_classes, input_tensor):
        data_tensor = input_tensor
        self.filter_tensors.append(tf.compat.v1.Variable(np.random.random([motif_width,# filter width
                                                    ALPHABET_SIZE,# in channels
                                                    num_classes]),# out channels
                                    name='filter_{}_width_{}_classes'.format(
                                        motif_width,
                                        num_classes),
                                    dtype=tf.compat.v1.float64))
        conv = tf.compat.v1.nn.conv1d(data_tensor,
                            self.filter_tensors[-1],
                            padding='SAME', # keep the dimensions of output the same
                            stride=1, # look at each residue
                            name='convolution_{}_width_{}_classes'.format(
                                motif_width,
                                num_classes))
        output = tf.compat.v1.math.reduce_mean(conv, axis=1)

        # This is for examining features.
        self.cov_features = tf.compat.v1.math.argmax(self.filter_tensors[-1], axis=1)

        return output

    def build_model(self, label_width, hyperparam_pairs, regression, learning_rate=MODEL_LEARNING_RATE):
        input_tensor = tf.compat.v1.placeholder(shape=np.array((None, MAX_LENGTH, ALPHABET_SIZE)),
                                    dtype=tf.compat.v1.float64,
                                    name='input')
        labels_tensor = tf.compat.v1.placeholder(shape=(input_tensor.shape[0], label_width),
                                    dtype=tf.compat.v1.int32,
                                    name='labels')
        dropout_rate = tf.compat.v1.placeholder_with_default(tf.compat.v1.constant(DEFAULT_DROPOUT_RATE, dtype=tf.compat.v1.float64), shape=(), name='dropout_rate')
        #0-1 length of peptide followed by avg of each amino acid
        aa_counts =  tf.compat.v1.concat([tf.compat.v1.reshape(tf.compat.v1.reduce_sum(input_tensor, axis=[1,2]) / float(MAX_LENGTH), [-1, 1]), tf.compat.v1.reduce_mean(input_tensor, axis=1)], axis=1)
        #aa_counts = MAX_LENGTH * tf.concat([tf.reshape(tf.reduce_sum(input_tensor, axis=[1,2]) / float(MAX_LENGTH), [-1, 1]), tf.reduce_mean(input_tensor, axis=1)], axis=1)
        #aa_counts = tf.concat([tf.reshape(tf.reduce_sum(input_tensor, axis=[1,2]), [-1, 1]), tf.reduce_sum(input_tensor, axis=1)], axis=1)
        convs = []
        for hparam_pair in hyperparam_pairs:
            convs.append(self.make_convolution(hparam_pair[0], hparam_pair[1], input_tensor))
        classifiers_losses = []
        logits = []
        self.classifier_outputs = []
        for i, conv in enumerate(convs):
            features = tf.compat.v1.concat([conv,aa_counts], axis=1)
            x0 = tf.compat.v1.nn.dropout(tf.compat.v1.layers.dense(features,
                                            HIDDEN_LAYER_SIZE,
                                            activation=tf.compat.v1.nn.tanh),
                            rate=dropout_rate)
            x = x0
            for j in range(HIDDEN_LAYER_NUMBER):
                    x = tf.compat.v1.nn.dropout(tf.compat.v1.layers.dense(x,
                                                      HIDDEN_LAYER_SIZE,
                                                      activation=tf.compat.v1.nn.relu),
                                      rate=dropout_rate)
            # x is now the final layer
            logits = tf.compat.v1.layers.dense(x, label_width)
            sigmoid_logits = tf.compat.v1.nn.sigmoid(logits)
            self.classifier_outputs.append(sigmoid_logits)
            #classifiers_losses.append(tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_tensor,logits=logits))
            classifiers_losses.append(tf.compat.v1.keras.losses.binary_crossentropy( y_true=labels_tensor, y_pred=sigmoid_logits))
            # output is 2D: probabilities of 'has this property'/'does not have'
            # easier to compare logits with one-hot labels this way
        # Instead of learner NN model, here we use uncertainty minimization
        self.total_classifiers_loss = tf.compat.v1.reduce_sum(classifiers_losses) / float(len(classifiers_losses))
        # join classifiers
        labels = tf.compat.v1.concat([x[tf.compat.v1.newaxis, :, :] for x in self.classifier_outputs], axis=0)
        # get majority vote
        # get avg prediction, take mean, then compare
        votes = tf.compat.v1.math.round(tf.compat.v1.reduce_mean(labels, axis=0))
        # [0, 1] - [1, 0] = [-1, 1] -> abs sum is 2, so divide by 2
        FPR = tf.compat.v1.reduce_sum(tf.compat.v1.abs(votes - tf.compat.v1.cast(labels_tensor, tf.compat.v1.float64)))# / 2.0
        self.accuracy = 1 - FPR / tf.compat.v1.cast(tf.compat.v1.shape(labels_tensor)[0], tf.compat.v1.float64)
        self.classifier_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.total_classifiers_loss)
        
        # for interpretation - only use last model (most hyperparameters)
        # we're getting partial (via stop_gradients) of positive label probability wrt aa_counts.
        count_grads = tf.compat.v1.gradients(self.classifier_outputs[-1][:,0], aa_counts, stop_gradients=aa_counts)
        # sum over batches (0) and ys (len = 1, axis = 1)
        # should be left with gradient of length aa_counts
        self.count_grads = tf.compat.v1.reduce_sum(count_grads, axis=[0, 1])

    def build_calibration_graph(self, logits, label_width):
        calibrated_classifiers_losses = []
        # uncalibrated logits
        s = tf.compat.v1.placeholder(shape=np.array((None, label_width)),
                                    dtype=tf.compat.v1.float64,
                                    name='input')
        self.calibrated_classifier_outputs = []
        a = tf.compat.v1.Variable(0.2, trainable=True, name='a', constraint=lambda t: tf.compat.v1.clip_by_value(t, 0.001, 10000.), dtype=tf.compat.v1.float64)
        b = tf.compat.v1.Variable(0.2, trainable=True, name='b', constraint=lambda t: tf.compat.v1.clip_by_value(t, 0.001, 10000.), dtype=tf.compat.v1.float64)
        c = tf.compat.v1.Variable(1.1, trainable=True, name='c', dtype=tf.compat.v1.float64)
        # beta calibration on inputs
        term1 = tf.compat.v1.math.exp(c) * tf.compat.v1.math.pow(softmax_logits, a)
        term2 = tf.compat.v1.math.pow((1. - softmax_logits), b)
        d = tf.compat.v1.ones_like(softmax_logits, dtype=tf.compat.v1.float64)
        calibrated_logits = tf.compat.v1.math.divide_no_nan(d, (d + tf.compat.v1.math.divide_no_nan(d, tf.compat.v1.math.divide_no_nan( term1, term2 ))))
        self.calibrated_classifier_outputs.append(calibrated_logits)#(tf.compat.v1.nn.softmax(calibrated_logits))
        self.total_calibrated_classifiers_loss = tf.compat.v1.reduce_sum(calibrated_classifiers_losses) / float(len(calibrated_classifiers_losses))
        self.calibrated_classifier_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate * 0.1).minimize(self.total_calibrated_classifiers_loss, var_list=[a,b,c])
        calibrated_classifiers_losses.append(tf.compat.v1.losses.log_loss(labels=labels_tensor,predictions=calibrated_logits))


    def train(self, sess, labels, peps, iters=DEFAULT_TRAIN_ITERS, batch_size=16, replacement=False):
        losses = [0 for _ in range(iters)]
        # only go as many iters as we have data (heuristic)
        iters = min(iters, peps.shape[0])
        for i in range(iters):
            # prevent not having unique elements by minning batch size with peps.shape
            indices = np.random.choice(peps.shape[0], min(peps.shape[0], batch_size), replace=replacement)
            # train with dropout
            #losses[i], _ = sess.run(
            #    [self.total_classifiers_loss, self.classifier_optimizer],
            #    feed_dict={'input:0': peps[indices], 'labels:0': labels[indices]})
            losses[i], _= sess.run([self.total_classifiers_loss, 
                                    self.classifier_optimizer], 
                                    feed_dict={'input:0': peps[indices], 'labels:0': labels[indices]})
        return losses

    def eval_loss(self, sess, labels, peps):
        return self.eval(sess, labels, peps)[0]

    def eval_accuracy(self, sess, labels, peps):
        return sess.run(self.accuracy,
            feed_dict={'input:0': peps, 'labels:0':labels, 'dropout_rate:0': 0.0})

    def eval_labels(self, sess, peps):
        retval = sess.run([self.classifier_outputs],
            feed_dict={'input:0': peps, 'dropout_rate:0': 0.0})
        return retval

    def eval(self, sess, labels, peps):
        return sess.run(
            [self.total_calibrated_classifiers_loss],
            feed_dict={'input:0': peps, 'labels:0': labels, 'dropout_rate:0': 0.0})

    def eval_motifs(self, sess):
        return sess.run(self.cov_features)

    def eval_count_grad(self, sess, peps):
        return sess.run(
            self.count_grads,
            feed_dict={'input:0': peps, 'dropout_rate:0': 0.0})



def mix_split(peps, labels, withheld_percent=0.2):
    all_peps = np.concatenate(peps, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    stop_idx = int(len(all_peps) * (1 - withheld_percent))
    train_peps = all_peps[:stop_idx]
    withheld_peps = all_peps[stop_idx:]
    train_labels = all_labels[:stop_idx]
    withheld_labels = all_labels[stop_idx:]
    shuffle_same_way([train_peps, train_labels])
    shuffle_same_way([withheld_peps, withheld_labels])
    return (train_labels, train_peps), (withheld_labels, withheld_peps)

def prepare_data(positive_filename, negative_filenames, regression, withheld_percent=0.2, weights = None):
    # set-up weights
    # not used yet
    # load randomly shuffled training data
    if regression:
        raise NotImplementedError()
        positive_peps, positive_withheld_peps, labels, withheld_labels = load_data(positive_filename, regression, negative_filename, withheld_percent=withheld_percent)
    # negative dataset only needed if we're classifying
    else:
        positive_peps, positive_withheld_peps = load_data(positive_filename, regression, withheld_percent=withheld_percent)
        if type(negative_filenames) != list:
            negative_filenames = [negative_filenames]
        negative_peps, negative_withheld_peps = [], []
        for n, w in negative_filenames:
            loaded = load_data(n, regression, withheld_percent=withheld_percent)
            negative_peps.extend(loaded[0])
            negative_withheld_peps.extend(loaded[1])
        # now downsample without replacement
        if len(negative_peps) < len(positive_peps):
            print('Unable to find enough negative examples for {}'.format(positive_filename))
            print('Using', *[n for n,w in negative_filenames])
            exit(1)
        negative_peps = random.sample(negative_peps, k=len(positive_peps))
        negative_withheld_peps = random.sample(negative_withheld_peps, k=len(positive_withheld_peps))
    # convert negative peps into array
    # I still don't get numpy...
    # why do I have to create a newaxis here? Isn't there an easier way?
    negative_peps = np.concatenate([n[np.newaxis, :, :] for n in negative_peps], axis=0)
    peps = np.concatenate([positive_peps, negative_peps]) if not regression else positive_peps
    withheld_peps = np.concatenate([positive_withheld_peps, negative_withheld_peps]) if not regression else positive_withheld_peps

    # calculate activities if we're not doing regression
    if not regression:
        labels = np.zeros(len(peps)) # one-hot encoding of labels
        labels[:len(positive_peps)] += 1. # positive labels are 1
        withheld_labels = np.zeros(len(withheld_peps)) # one-hot encoding of labels
        withheld_labels[:len(positive_withheld_peps)] += 1. # positive labels are 1

    # now re-shuffle all these in the same way
    shuffle_same_way([labels, peps])
    shuffle_same_way([withheld_labels, withheld_peps])
    return (labels, peps), (withheld_labels, withheld_peps)


def load_datasets(root, withheld_percent=0.2):
    with open(os.path.join(root, 'dataset_names.txt')) as f:
        dataset_names = f.readlines()
    # trim whitespace
    dataset_names = [n.split()[0] for n in dataset_names]
    dataset_fakes = {n: [] for n in dataset_names}
    with open(os.path.join(root, 'dataset_fakes.txt')) as f:
        for line in f.readlines():
            sline = line.split()
            if sline[0][0] == '#':
                continue
            n = sline[0]
            index = 1
            while index < len(sline):
                fn = sline[index]
                if fn.find('fake') != -1:
                    dataset_fakes[n].append(('{}-fake'.format(n), 'x'))
                    break
                dataset_fakes[n].append((sline[index], sline[index + 1]))
                index += 2
    datasets = []
    for n in dataset_names:
        pos = os.path.join(root, '{}-sequence-vectors.npy'.format(n))
        negs = []
        for nf, nw in dataset_fakes[n]:
            negs.append((os.path.join(root, '{}-sequence-vectors.npy'.format(nf)), nw))
        train, withheld = prepare_data(pos, negs, False, withheld_percent=withheld_percent)
        datasets.append((n, train, withheld))
    return datasets

def project_peptides(name, seqs, weights, cmap=None, labels=None, ax=None, colorbar=True):
    import sklearn.manifold
    import sklearn.decomposition
    import umap
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    if type(weights[0]) != list:
        weights = [weights]
    flat = np.reshape(seqs, (seqs.shape[0], -1))
    assert flat.shape[0] == len(weights[0])
    pca = sklearn.decomposition.PCA(40)
    seqs_pca = pca.fit_transform(flat)
    features = seqs_pca
    embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(features)
    #embedded = umap.UMAP(min_dist=0.1,  n_neighbors=8).fit_transform(flat, y=weights[0])
    #embedded = umap.UMAP(min_dist=0.1,  n_neighbors=8).fit_transform(flat)

    for i,w in enumerate(weights):
        if ax is None:
            plt.figure(figsize=(7,4))
            _ax = plt.gca()
        else:
            _ax = ax
        if name is not None:
            _ax.set_title(name)
        if len(seqs) > 2000:
            s = 1.0
        else:
            s = 3
        sc = _ax.scatter(embedded[:, 0], embedded[:,1], c=w, s=s, edgecolors='face', linewidth=0.0, cmap=cmap, alpha=0.8)
        plt.setp(_ax, xticks=[], yticks=[])
        if colorbar:
            if labels is None:
                plt.colorbar()
            else:
                uw = np.sort(np.unique(w))
                cbar = plt.colorbar(boundaries=np.arange(len(labels))-0.5)
                cbar.set_ticks(np.arange(len(labels)))
                cbar.set_ticklabels(labels)
        if ax is None:
            plt.tight_layout()
            plt.savefig('{}-{}-projection.png'.format(name, i), dpi=300)

def evaluate_strategy(train_data, withheld_data, learner, output_dirname, strategy=None,
                      nruns=10, index=0, regression=False, sess=None, plot_umap=False, batch_size=16):
    peps = train_data[1]
    labels = train_data[0][:, None]
    withheld_peps = withheld_data[1]
    withheld_labels = withheld_data[0][:, None]

    train_losses = []
    withheld_accuracy = []
    pep_choice_indices = []

    with tf.compat.v1.Session() if sess is None else sess.as_default() as _sess:
        # here is where the sessions are set up and called
        if sess is None:
            _sess.run(tf.compat.v1.global_variables_initializer())
        sess = _sess
        # do get initial loss
        withheld_accuracy.append(learner.eval_accuracy(sess, withheld_labels, withheld_peps))
        for i in range(nruns):
            # run the classifiers on all available peptides to get their outputs
            # then pick the one according to the strategy
            output = learner.eval_labels(sess, peps)
            # make random selections for next training point.
            if strategy is None:
                train_losses.append(learner.train(sess, labels, peps, batch_size=batch_size, iters=nruns)[-1])
                withheld_accuracy.append(learner.eval_accuracy(sess, withheld_labels, withheld_peps))
                break
            chosen_idx = strategy(peps, output, regression)
            pep_choice_indices.append(chosen_idx)
            # train for the chosen number of steps after each observation
            # only append final training value
            train_losses.append(learner.train(sess, labels[pep_choice_indices], peps[pep_choice_indices], batch_size=batch_size)[-1])
            withheld_accuracy.append(learner.eval_accuracy(sess, withheld_labels, withheld_peps))

        # now that training is done, get final withheld predictions
        final_withheld_predictions = learner.eval_labels(sess, withheld_peps)
        final_train_predictions = learner.eval_labels(sess, peps)
        motifs = learner.eval_motifs(sess)
        # make poly-alanine
        polyalanine = np.zeros((1,MAX_LENGTH, ALPHABET_SIZE))
        polyalanine[0, :10, 0] = 1.0
        count_grads = learner.eval_count_grad(sess, polyalanine)

    if plot_umap:
        project_peptides(os.path.join(output_dirname, str(index)), peps,  [final_train_predictions[0][:,1],  1. - labels])
    index = str(index) # make sure it's a string
    np.savetxt('{}/{}_train_losses.txt'.format(output_dirname, index.zfill(4)), train_losses)
    np.savetxt('{}/{}_withheld_accuracy.txt'.format(output_dirname, index.zfill(4)), withheld_accuracy)
    np.savetxt('{}/{}_choices.txt'.format(output_dirname, index.zfill(4)), pep_choice_indices)
    np.savetxt('{}/{}_motifs.txt'.format(output_dirname, index.zfill(4)), motifs)
    np.savetxt('{}/{}_count_grads.txt'.format(output_dirname, index.zfill(4)), count_grads)


    # can't do ROC for regression
    if not regression and nruns > 0:
        # AUC analysis (misclassification) for final withheld predictions
        # iterate over all models
        if len(final_withheld_predictions) == 1:
            #not qbc
            predictions_arr = final_withheld_predictions[0]
        else:
            #in qbc, do average of committee predictions for AUC
            predictions_arrs = []
            for predictions_arr in final_withheld_predictions:
                predictions_arrs.append(predictions_arr)
            predictions_arr = np.mean(np.array(predictions_arrs), axis=0)
        withheld_fpr, withheld_tpr, withheld_threshold = roc_curve(withheld_labels,
                                                                   predictions_arr[0])
        np.save('{}/{}_fpr.npy'.format(output_dirname, index.zfill(4)), withheld_fpr)
        np.save('{}/{}_tpr.npy'.format(output_dirname, index.zfill(4)), withheld_tpr)
        np.save('{}/{}_thresholds.npy'.format(output_dirname, index.zfill(4)), withheld_threshold)
        withheld_auc = auc(withheld_fpr, withheld_tpr)
        np.savetxt('{}/{}_auc.txt'.format(output_dirname, index.zfill(4)), [withheld_auc])
    # for regression, instead rank the training set and record results.
    elif regression:
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
