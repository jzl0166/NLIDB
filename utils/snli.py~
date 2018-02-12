import os
import json

import numpy as np


def load_data(snli='~/data/snli', embedding=None):

    def _read(fname):
        print('\nReading {}'.format(fname))
        X, y = [], []
        with open(fname, 'r') as f:
            for line in f:
                dat = json.loads(line)

                label = dat['gold_label']
                if '-' == label:
                    continue;

                X.append([dat['sentence1'], dat['sentence2']])
                y.append(dat['gold_label'])
        return X, y

    fname = os.path.join(snli, 'snli_1.0_train.jsonl')
    X_train, y_train = _read(fname)

    fname = os.path.join(snli, 'snli_1.0_dev.jsonl')
    X_dev, y_dev = _read(fname)

    fname = os.path.join(snli, 'snli_1.0_test.jsonl')
    X_test, y_test = _read(fname)

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data('snli')

    print('X_train len: {}'.format(len(X_train)))
    print('X_dev len: {}'.format(len(X_dev)))
    print('X_test len: {}'.format(len(X_test)))
