import unittest

import numpy as np

from tensorslow.optimizers import SGD


class TestOptimizers(unittest.TestCase):

    def setUp(self) -> None:

        self.current_weights = [np.array([[3, 6], [8, 4], [2, 5]]), np.array([1, 2])[None, :]]  # weights, bias
        self.weight_gradients = self.current_weights
        self.lr = 1

    def test_sgd_get_updated_weights(self):
        sgd = SGD(lr=self.lr)
        updated_weights = sgd.get_updated_weights(self.current_weights, self.weight_gradients)
        np.testing.assert_equal(updated_weights[0], np.zeros(self.current_weights[0].shape))
        np.testing.assert_equal(updated_weights[1], np.zeros(self.current_weights[1].shape))


if __name__ == '__main__':
    unittest.main()