import numpy as np
import pickle
from sys import argv
import os
import tensorflow as tf
from utils import *
import tqdm
from active_learn import get_active_learner, evaluate_strategy

def inner_iter(sess, learner, k, labels, peps, strategy, swap_labels=True):
    pep_choice_indices = []
    if swap_labels and np.random.uniform() < 0.5:
        labels[:,0] = 1 - labels[:,1]
        labels[:,1] = 1 - labels[:,0]
    output = learner.eval_labels(sess, peps)
    for i in range(k):
        if strategy is None:
            loss = learner.train(sess, labels, peps, iters=5)[-1]
        else:
            chosen_idx = strategy(peps, output, False)
            pep_choice_indices.append(chosen_idx)
            # train for the chosen number of steps after each observation
            # only append final training value
            loss = learner.train(sess, labels[pep_choice_indices], peps[pep_choice_indices], iters=5)[-1]
    return loss

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print('reptile.py [data_root] [output_root] [withhold-index] [strategy]')
        exit()
    root = sys.argv[1]
    withhold_index = int(sys.argv[3])
    strategy_str = sys.argv[4]
    output_dirname = '{}/{}/{}'.format(sys.argv[2], strategy_str, withhold_index)
    os.makedirs(output_dirname, exist_ok=True)
    LEARNING_RATE = 0.01
    META_TRAIN_ITERS = 1500
    META_PERIOD = 25
    META_INNER_SAMPLES = 5
    META_VALIDATION_SAMPLES = 10
    eta = LEARNING_RATE
    # get data names
    with open(os.path.join(root, 'dataset_names.txt')) as f:
        dataset_names = f.readlines()
    # trim whitespace
    dataset_names = [n.split()[0] for n in dataset_names]
    datasets = []
    for n in dataset_names:
        positive_filename = os.path.join(root, '{}-sequence-vectors.npy'.format(n))
        negative_filename = os.path.join(root, '{}-fake-sequence-vectors.npy'.format(n))
        (labels, peps), (withheld_labels, withheld_peps) = prepare_data(positive_filename, negative_filename, False, withheld_percent=0)
        datasets.append([n, labels, peps])

    # withhold one dataset
    print('Withholding {}'.format(datasets[withhold_index][0]))
    strategy, hyperparam_pairs = get_active_learner(strategy_str)
    learner = Learner(labels.shape[1], hyperparam_pairs, False)
    # get trainables from learner + strategy
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
            inner_iter(sess, learner, META_INNER_SAMPLES, data[1], data[2], strategy)
            sess.run(update_hypers)
            sess.run(reset_vars)
            # eval on withheld dataset
            data =  datasets[withhold_index]
            losses = inner_iter(sess, learner, META_INNER_SAMPLES, data[1], data[2], strategy)
            previous_losses.append(losses)
            del previous_losses[0]
            # reset so we don't save the training
            sess.run(reset_vars)
            if global_step % META_PERIOD == 0:
                saver.save(sess, output_dirname + '/model', global_step=global_step)
                # check if still making progress
                pmean = sum(previous_losses) / META_PERIOD
                print('Mean validation loss on last period is', pmean)
                if  pmean < best:
                    best = pmean
                    print('New mean best is {}'.format(best))
                elif global_step > META_PERIOD * 5:
                    print('Failing to make progress')
                    break
        # get final training
        data =  datasets[withhold_index]
        loss = inner_iter(sess, learner, META_INNER_SAMPLES, data[1], data[2], strategy)
        print('Loss on validation task {} is {}'.format(data[0], loss))
        sess.run(reset_vars)
        print('running evaluation')
        # need to reload in order to split validation data
        n = dataset_names[withhold_index]
        positive_filename = os.path.join(root, '{}-sequence-vectors.npy'.format(n))
        negative_filename = os.path.join(root, '{}-fake-sequence-vectors.npy'.format(n))
        (labels, peps), (withheld_labels, withheld_peps) = prepare_data(positive_filename, negative_filename, False)
        for i in tqdm.tqdm(range(META_VALIDATION_SAMPLES)):
            sess.run(reset_vars)
            evaluate_strategy((labels, peps), (withheld_labels, withheld_peps), learner,
                output_dirname, strategy=strategy, index=i, regression=False, sess=sess)
