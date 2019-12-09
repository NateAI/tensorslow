""" Matrix opterations"""

import numpy as np


def batch_matmul(a, b):
    """
    multiply all matrices in batch
    Parameters
    ----------
    a: np.ndarray
        [batch_size, a_rows , a_cols]
    b: np.ndarray
        [batch_size, b_rows, b_cols]

    Returns
    -------
    result: np.ndarray
        [batch_size, a_rows, b_cols]
    """
    assert np.ndim(a) == 3, 'batch_matmul requires tensors of rank 3'
    assert np.ndim(b) == 3,  'batch_matmul requires tensors of rank 3'
    assert a.shape[0] == b.shape[0], 'batch size different for each tensor'

    batch_size = a.shape[0]
    result = np.zeros(batch_size, a.shape[1], b.shape[2])
    for example_idx in range(batch_size):
        result[example_idx] = np.matmul(a[example_idx], b[example_idx])

    return result