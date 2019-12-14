import unittest

import numpy as np

from tensorslow.layers import CategoricalCrossentropy


class TestLossFunctions(unittest.TestCase):

    def setUp(self) -> None:

        self.y_true = np.array([[0, 1], [1, 0], [0, 1]])
        self.y_pred = np.array([[0.5, 0.5], [0.9, 0.1], [0.8, 0.2]])
        self.tolerance = 1e-6

        self.cce = CategoricalCrossentropy()

    def test_categorical_crossentropy_forward_pass(self):

        losses = self.cce.forward_pass(self.y_true, self.y_pred)
        desired = np.array([-np.math.log(0.5), -np.math.log(0.9), -np.math.log(0.2)])
        desired = np.expand_dims(desired, axis=1)
        np.testing.assert_allclose(losses, desired, atol=self.tolerance,
                                   err_msg='CategoricalCrossentropy forward_pass unit test failed')

    def test_categorical_crossentropy_backward_pass(self):

        # TODO figure out why the test fails if I don't repeat the line below???
        losses = self.cce.forward_pass(self.y_true, self.y_pred)
        gradients = self.cce.backward_pass()
        desired = np.array([[0, - 1/0.5], [- 1/0.9, 0], [0, - 1/0.2]])
        np.testing.assert_allclose(gradients, desired, atol=self.tolerance,
                                   err_msg='CategoricalCrossentropy backward_pass unit test failed')


if __name__ == '__main__':
    unittest.main()