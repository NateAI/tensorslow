
import numpy as np

from tensorslow.layers.layer import Layer


class Sigmoid(Layer):

    def __init__(self):

        super(Sigmoid, self).__init__()

    def forward_pass(self, prev_layer_output):
        """ σ(z) = 1 / (1 + e^(-z))"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        return 1 / (1 + np.exp(-1 * prev_layer_output))


class Softmax(Layer):

    def __init__(self):

        super(Softmax, self).__init__()

    def forward_pass(self, prev_layer_output):
        """ σ(z_i) =  e^(z_i) / sum({e^(z_j) for all j}) """

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        return np.exp(prev_layer_output) / np.sum(np.exp(prev_layer_output), axis=1)[:, None]


class Relu(Layer):

    def __init__(self):

        super(Relu, self).__init__()

    def forward_pass(self, prev_layer_output):
        """ σ(z) = max(0, z)"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        return np.maximum(prev_layer_output, np.zeros(shape=prev_layer_output.shape))


class Tanh(Layer):

    def __init__(self):

        super(Tanh, self).__init__()

    def forward_pass(self, prev_layer_output):
        """ σ(z) = (e^z - e^-z) / (e^z + e^-z)"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        e_z = np.exp(prev_layer_output)
        e_minus_z = np.exp(-1 * prev_layer_output)

        return (e_z - e_minus_z) / (e_z + e_minus_z)