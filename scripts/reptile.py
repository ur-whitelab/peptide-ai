import numpy as np
import pickle
from sys import argv
import os
import tensorflow as tf
from utils import *
import tqdm
from active_learn import get_active_learner, evaluate_strategy

META_LEARNING_RATE = 1e-4
META_TRAIN_ITERS = 2000
META_PERIOD = 50
META_INNER_SAMPLES = 5
META_VALIDATION_SAMPLES = 30
META_VALIDATION_LENGTH = 25
LABEL_DIMENSION = 2
TEST_ZERO_SHOT = True
SWAP_LABELS = True


def inner_iter(sess, learner, k, labels, peps, strategy, swap_labels=SWAP_LABELS):
    pep_choice_indices = []
    if swap_labels and np.random.uniform() < 0.5:
        swapped_labels = np.zeros_like(labels)
        swapped_labels[:,0] = 1 - labels[:,0]
        swapped_labels[:,1] = 1 - labels[:,1]
        labels = swapped_labels
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
    eta = META_LEARNING_RATE
    output_dirname = '{}/{}/{}'.format(sys.argv[2], strategy_str, withhold_index)
    os.makedirs(output_dirname, exist_ok=True)
    # get data names
    datasets = load_datasets(root)
    # withhold one dataset
    print('Withholding {}'.format(datasets[withhold_index][0]))
    strategy, hyperparam_pairs = get_active_learner(strategy_str, stochastic=False)
    learner = Learner(LABEL_DIMENSION, hyperparam_pairs, False, MODEL_LEARNING_RATE)
    # get trainables from learner + strategy
    to_train = tf.trainable_variables()

    # now create hyper version
    hyper_trains = [tf.get_variable(shape=v.shape, dtype=v.dtype, name='hyper-{}'.format(v.name.split(':')[0])) for  v in to_train]
    update_hypers = tf.train.AdamOptimizer(eta).apply_gradients([(h - v, h) for h,v in zip(hyper_trains, to_train)])
    #update_hypers = tf.group(*[h.assign(v * eta + (1 - eta) * h) for h,v in zip(hyper_trains, to_train)])
    reset_vars = tf.group(*[v.assign(h) for h,v in zip(hyper_trains, to_train)])
    var_delta = tf.reduce_sum([tf.reduce_sum((h - v)**2) for h,v in zip(hyper_trains, to_train)])

    saver = tf.train.Saver(hyper_trains)
    with tf.Session() as sess:
        # init
        sess.run(tf.global_variables_initializer())

        previous_losses = [np.nan] * META_PERIOD
        best = 100
        # set to hypers
        sess.run(reset_vars)
        # training loop
        with tqdm.tqdm(total=META_TRAIN_ITERS) as pbar:
            for global_step in range(META_TRAIN_ITERS):
                # sample dataset
                while True:
                    data_index = np.random.randint(0, len(datasets))
                    if data_index != withhold_index:
                        break
                _name, data, _withheld =  datasets[data_index]
                inner_iter(sess, learner, META_INNER_SAMPLES, data[0], data[1], strategy)
                update_size = sess.run(var_delta)
                sess.run(update_hypers)
                sess.run(reset_vars)
                # eval on withheld dataset
                _name, data, _withheld =  datasets[withhold_index]
                losses = inner_iter(sess, learner, META_INNER_SAMPLES, data[0], data[1], strategy)
                previous_losses.append(losses)
                del previous_losses[0]
                # reset so we don't save the training
                sess.run(reset_vars)
                pbar.update(1)
                pbar.set_description('Dataset: {}, Change: {:.5}, Loss: {:.2f}'.format(data_index, update_size, np.nanmean(previous_losses)))
                if global_step % META_PERIOD == 0:
                    saver.save(sess, output_dirname + '/model', global_step=global_step)
                    # check if still making progress
                    pmean = np.nanmean(previous_losses)
                    if  pmean < best:
                        best = pmean
                    elif global_step > META_TRAIN_ITERS * 0.5:
                        print('Failing to make progress')
                        break
        # get final training
        data_name, data, _withheld =  datasets[withhold_index]
        loss = inner_iter(sess, learner, META_INNER_SAMPLES, data[0], data[1], strategy)
        print('Loss on validation task {} is {}'.format(data_name, loss))
        sess.run(reset_vars)
        print('running evaluation')
        # zero shot
        if TEST_ZERO_SHOT:
            os.makedirs(output_dirname + 'zero', exist_ok=True)
            evaluate_strategy(datasets[withhold_index][1], datasets[withhold_index][2], learner,
                    output_dirname + 'zero', strategy=strategy, index=0, nruns=0, regression=False, sess=sess, plot_umap=False)
            # zero shot with swapped labels
            if SWAP_LABELS:
                for i in range(1,3): #iter over train/withheld
                        for j in range(2): #iter over class
                            datasets[withhold_index][i][0][:,j] = 1 - datasets[withhold_index][i][0][:,j]
            evaluate_strategy(datasets[withhold_index][1], datasets[withhold_index][2], learner,
                    output_dirname + 'zero', strategy=strategy, index=1, nruns=0, regression=False, sess=sess, plot_umap=False)
        for index in tqdm.tqdm(range(META_VALIDATION_SAMPLES)):
            sess.run(reset_vars)
            (labels, peps), (withheld_labels, withheld_peps) = mix_split([datasets[withhold_index][1][1], datasets[withhold_index][2][1]],
                                                                         [datasets[withhold_index][1][0], datasets[withhold_index][2][0]])
            # swap labels around
            if SWAP_LABELS:
                for i in range(1,3): #iter over train/withheld
                    for j in range(2): #iter over class
                        #        dataset index t/v lab
                        datasets[withhold_index][i][0][:,j] = 1 - datasets[withhold_index][i][0][:,j]
            evaluate_strategy(datasets[withhold_index][1], datasets[withhold_index][2], learner,
                output_dirname, strategy=strategy, index=index, nruns=META_VALIDATION_LENGTH, regression=False, sess=sess)
