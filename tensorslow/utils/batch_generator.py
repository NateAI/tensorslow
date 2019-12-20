
import numpy as np
from tqdm import tqdm


def batch_generator(x, y, batch_size, shuffle=True):

    verify_batch_generator_input(x, y, batch_size, shuffle=True)

    num_examples = x.shape[0]

    if shuffle:
        perm = np.random.permutation(num_examples)
        x = x[perm]
        y = y[perm]

    num_batches_per_epoch = np.math.ceil(num_examples / batch_size)
    remainder = num_examples % batch_size  # final batch of each epoch may be of smaller size

    for batch_idx in tqdm(range(num_batches_per_epoch)):

        if batch_idx == (num_batches_per_epoch - 1) and remainder > 0:
            x_batch = x[batch_size * batch_idx:]  # if last batch and batch_size does not divide num_examples
            y_batch = y[batch_size * batch_idx:]  # then use all remaining examples in final batch
        else:
            x_batch = x[batch_size * batch_idx: batch_size * (batch_idx + 1)]  # typical case
            y_batch = y[batch_size * batch_idx: batch_size * (batch_idx + 1)]

        yield x_batch, y_batch


def verify_batch_generator_input(x, y, batch_size, shuffle=True):

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert np.ndim(x) >= 2
    assert np.ndim(y) >= 2
    assert x.shape[0] == y.shape[0]  # check num examples is equal
    assert isinstance(batch_size, int)
    assert 0 < batch_size < x.shape[0], 'batch_size parameter must be in range[0, num_examples]'
    assert isinstance(shuffle, bool)



