""" Implement metrics to be compute on evaluation"""

import numpy as np


def accuracy(y_true, y_pred):

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return np.sum(y_pred == y_true) / len(y_true)