import numpy as np

from tensorslow.layers.layer import ParametricLayer


class FullyConnected(ParametricLayer):

    def __init__(self, neurons, input_dim):

        super(FullyConnected, self).__init__(neurons, input_dim)

    def forward_pass(self, prev_layer_output):
        """ f(x) = W*X + b"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        return np.matmul(prev_layer_output, self._weights) + self._bias


