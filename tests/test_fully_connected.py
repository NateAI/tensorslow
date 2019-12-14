import unittest

import numpy as np

from tensorslow.layers import FullyConnected


class TestFullyConnected(unittest.TestCase):

    def setUp(self) -> None:

        self.batch_size = 2
        self.input_dim = 3
        self.neurons = 2

        self.x = np.array([[-1, 0, 1],
                           [1, 0, -1]])

        self.fully_connected = FullyConnected(neurons=self.neurons, input_dim=self.input_dim)
        self.fully_connected.weights = np.array([[2, 3],
                                                 [1, 0],
                                                 [3, 2]])
        self.fully_connected.bias = np.array([1, 2])[None, :]

        self.next_layer_gradients = np.array([[0, 1],
                                              [2, -1]])

        self.tolerance = 1e-5

    def test_fully_connected_forward_pass(self):

        logits = self.fully_connected.forward_pass(self.x)

        desired = np.array([[2, 1],
                            [0, 3]])

        np.testing.assert_equal(logits, desired, err_msg='Failed to calculate fully-connected forward pass correctly')

    def test_fully_connected_weight_gradients(self):
        """Test calculation of partial derivatives of loss wrt weights"""
        logits = self.fully_connected.forward_pass(self.x)
        gradients = self.fully_connected.get_weight_gradients(self.next_layer_gradients)

        # Test by comparing with less efficient but less error prone method of calculating
        desired = np.zeros(shape=(self.batch_size, self.input_dim, self.neurons))
        for batch_idx in range(self.batch_size):
            for column in range(self.neurons):
                for row in range(self.input_dim):
                    desired[batch_idx, row, column] = self.x[batch_idx, row] * self.next_layer_gradients[batch_idx, column]

        desired = np.mean(desired, axis=0)

        np.testing.assert_equal(gradients, desired, err_msg='Failed to calculate fully-connected partial derivatives wrt weights')

    def test_fully_connected_bias_gradients(self):

        logits = self.fully_connected.forward_pass(self.x)
        gradients = self.fully_connected.get_bias_gradients(self.next_layer_gradients)

        desired = np.array([1, 0])[None, :]

        np.testing.assert_allclose(gradients, desired, atol=self.tolerance)

    def test_fully_connected_backward_pass(self):

        logits = self.fully_connected.forward_pass(self.x)
        gradients = self.fully_connected.backward_pass(self.next_layer_gradients)

        # Calculate partial derivatives of loss wrt input (x) in a less efficient but less error prone way
        desired = np.zeros(shape=(self.batch_size, self.input_dim))
        for batch_idx in range(self.batch_size):
            for i in range(self.input_dim):
                desired[batch_idx, i] = np.sum([self.fully_connected.weights[i, 0] * self.next_layer_gradients[batch_idx, 0],
                                                 self.fully_connected.weights[i, 1] * self.next_layer_gradients[batch_idx, 1]])

        np.testing.assert_allclose(gradients, desired, atol=self.tolerance)




