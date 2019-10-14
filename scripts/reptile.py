import numpy as np
import pickle
from sys import argv
import os
import tensorflow as tf
from utils import *

'''This script takes in a directory that should be full of vectorized peptides,
   which can be created from raw APD files via vectorize_peptides.py
   It pads all peptide sequences out to a max length of 200, which is taken
   from observing that the max length of a peptide in the APD dataset is 183.
   Padding recommended by TF devs:
   https://github.com/tensorflow/graphics/issues/2#issuecomment-497428806'''



if __name__ == '__main__':
    import sys 
    if len(sys.argv) < 4:
        print('reptile.py [data_root] [output] [withhold-index]')
        exit()
    root = sys.argv[1]
    output_dirname = sys.argv[2]
    withhold_index = int(sys.argv[3])
    LEARNING_RATE = 0.01
    META_TRAIN_ITERS = 1500
    META_PERIOD = 25
    eta = LEARNING_RATE
    # get data names
    with open(os.path.join(root, 'dataset_names.txt')) as f:
        names = f.readlines()
    # trim whitespace
    names = [n.split()[0] for n in names]
    datasets = []
    for n in names:
        positive_filename = os.path.join(root, '{}-sequence-vectors.npy'.format(n))
        negative_filename = os.path.join(root, '{}-fake-sequence-vectors.npy'.format(n))
        (labels, peps), (withheld_labels, withheld_peps) = prepare_data(positive_filename, negative_filename, False, withheld_percent=0)
        datasets.append([n, labels, peps])

    # withhold one dataset
    print('Withholding {}'.format(datasets[withhold_index][0]))
    hyperparam_pairs = [(5, 6)]
    learner = Learner(labels.shape[1], hyperparam_pairs, False)
    # get trainables from learner
    to_train = tf.trainable_variables()

    # now create hyper version
    hyper_trains = [tf.get_variable(shape=v.shape, dtype=v.dtype, name='hyper-{}'.format(v.name.split(':')[0])) for  v in to_train]
    print(hyper_trains)
    update_hypers = tf.group(*[h.assign(v * eta + (1 - eta) * v) for h,v in zip(hyper_trains, to_train)])
    reset_vars = tf.group(*[v.assign(h) for h,v in zip(hyper_trains, to_train)])

    saver = tf.train.Saver(hyper_trains)
    with tf.Session() as sess:
        # init
        sess.run(tf.global_variables_initializer())

        previous_losses = [100] * META_PERIOD
        best = 100
        # set to hypers
        sess.run(reset_vars)
        # training loop
        for global_step in range(META_TRAIN_ITERS):
            # sample dataset
            while True:
                data_index = np.random.randint(0, len(datasets))
                if data_index != withhold_index:
                    break
            data =  datasets[data_index]
            learner.train(sess, data[1], data[2])
            sess.run(update_hypers)
            sess.run(reset_vars)
            # eval on withheld
            data =  datasets[withhold_index]
            losses = learner.train(sess, data[1], data[2])
            previous_losses.append(losses[0])
            del losses[0]
            # reset so we don't save the training
            sess.run(reset_vars)
            if global_step % META_PERIOD == 0:
                print('Loss on task {} is {}'.format(data[0], losses[-1]))
                saver.save(sess, output_dirname + '/{}/model'.format(withhold_index), global_step=global_step)
                # check if still making progress
                if min(previous_losses) < best:
                    best = min(previous_losses)
                    print('New best is {}'.format(best))
                elif global_step > 250:
                    print('Failing to make progress')
                    break
        # get final training
        data =  datasets[withhold_index]
        choice = np.random.choice(data[1].shape[0], 5, replace=False)
        losses = learner.train(sess, data[1][choice], data[2][choice])
        print('Loss on validation task {} is {}'.format(data[0], losses[-1]))
