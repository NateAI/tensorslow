import unittest

import numpy as np

from tensorslow.layers import Sigmoid, Softmax


class TestActivationFunctions(unittest.TestCase):

    def setUp(self) -> None:

        self.logits = np.array([[-1, 5, 0], [0, 5, -1]])
        self.next_layer_gradients = np.array([[0, 1, -1], [0, 1, -1]])

        self.softmax = Softmax()
        self.sigmoid = Sigmoid()

        self.tolerance = 1e-5

    def test_sigmoid_forward_pass(self):

        activations = self.sigmoid.forward_pass(self.logits)
        desired = np.array([[0.26894142, 0.99330714, 0.5],
                            [0.5, 0.99330714, 0.26894142]])
        np.testing.assert_allclose(activations, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate sigmoid forward pass correctly')

    def test_sigmoid_gradient_calculation(self):
        """Test calculation of sigmoid Jacobian"""
        activations = self.sigmoid.forward_pass(self.logits)
        sigmoid_gradients = self.sigmoid.sigmoid_gradients()
        desired = activations * (1 - activations)
        np.testing.assert_allclose(sigmoid_gradients, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate sigmoid derivative correctly')

    def test_sigmoid_backward_pass(self):
        """Test calculation of partial derivatives wrt loss"""
        # Test calculation of partial derivities wrt loss
        activations = self.sigmoid.forward_pass(self.logits)
        gradients = self.sigmoid.backward_pass(self.next_layer_gradients)
        desired = (activations * (1 - activations)) *self.next_layer_gradients
        np.testing.assert_allclose(gradients, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate sigmoid partial derivatives wrt loss')

    def test_softmax_forward_pass(self):
        activations = self.softmax.forward_pass(self.logits)

        desired = np.array([[0.00245612, 0.9908675, 0.00667641], [0.00667641, 0.9908675, 0.00245612]])

        np.testing.assert_allclose(activations, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate softmax forward pass correctly')

    def test_softmax_gradient_calculation(self):
        """Test calculation of Softmax Jacobian"""
        activations = self.softmax.forward_pass(self.logits)
        softmax_gradient = self.softmax.softmax_gradients()

        # Calculate Jacobian in longer but more risk free way
        batch_size = activations.shape[0]
        neurons = activations.shape[1]
        desired = np.zeros((batch_size, neurons, neurons))
        for batch_idx in range(batch_size):
            for i in range(neurons):
                for j in range(neurons):
                    p_i = activations[batch_idx, i]
                    p_j = activations[batch_idx, j]
                    kronecker = 1 if i == j else 0

                    desired[batch_idx, i, j] = p_i * (kronecker - p_j)

        np.testing.assert_allclose(softmax_gradient, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate softmax derivative correctly')

    def test_softmax_backward_pass(self):
        """ Test calculation of softmax partial derivatives wrt loss"""

        activations = self.softmax.forward_pass(self.logits)
        jacobian = self.softmax.softmax_gradients()

        gradients = self.softmax.backward_pass(self.next_layer_gradients)

        desired = np.squeeze(np.matmul(self.next_layer_gradients[:, None, :], jacobian))

        np.testing.assert_allclose(gradients, desired, atol=self.tolerance,
                                   err_msg='Failed to calculate partial derivatives of loss wrt softmax input')

if __name__ == '__main__':
    unittest.main()