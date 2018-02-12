from __future__ import print_function
from __future__ import division
"""
Convenient utilities, include training, evaluation and prediction.
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    print('\n Saving model')
    os.makedirs('model', exist_ok=True)
    env.saver.save(sess, 'model/{}'.format(name))


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        batch_y = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = batch_y
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128,
              fname=None):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={
            env.x: X_data[start:end],
            env.eps: eps, env.epochs: epochs})
        X_adv[start:end] = adv
    print()

    if fname is not None:
        np.save(fname, X_adv)

    return X_adv


def make_jsma(sess, env, X_data, y_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate JSMA by running env.x_jsma.
    """
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_jsma, feed_dict={
            env.x: X_data[start:end],
            env.y_target: y_data[start:end],
            env.eps: eps, env.epochs: epochs})
        X_adv[start:end] = adv
    print()
    return X_adv


def random_plot(X_data, X_advs, name='tmp.png'):
    ind = np.arange(X_data.shape[0])
    np.random.shuffle(ind)
    ind = ind[:10]

    imgs = X_data[ind]
    advs = X_advs[ind]

    fig = plt.figure(figsize=(10, 2.5))
    gs = gridspec.GridSpec(2, 10, wspace=0.05, hspace=0.05)

    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(imgs[i], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(advs[i], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/{}'.format(name))


def internal_repr(sess, env, X_data, batch_size=128):
    print('\nExtract internal representation')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    shape = [X_data.shape[0]] + env.z.get_shape().as_list()[1:]
    X_tmp = np.empty(shape)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tmp = sess.run(env.z, feed_dict={env.x: X_data[start:end]})
        X_tmp[start:end] = tmp
    print()
    return X_tmp
