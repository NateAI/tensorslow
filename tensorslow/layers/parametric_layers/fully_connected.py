import numpy as np

from tensorslow.layers.layer import ParametricLayer


class FullyConnected(ParametricLayer):

    def __init__(self, neurons, input_dim):

        super(FullyConnected, self).__init__(neurons, input_dim)

        self.prev_layer_output = None  # store on forward pass for backward pass
        self.logits = None

    def forward_pass(self, prev_layer_output):
        """ f(x) = W*X + b"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)
        self.prev_layer_output = prev_layer_output
        self.logits = np.matmul(prev_layer_output, self._weights) + self._bias

        return self.logits

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        """
        Compute partial derivatives of the output of this layer it's wrt input

        Parameters
        ----------
        next_layer_gradients: np.ndarray
            partial derivatives of loss wrt outputs of this layer [batch_size, 1, num_output_neurons]

        args
        kwargs

        Returns
        -------
        gradients: np.ndarray
            partial derivatives of loss wrt inputs of this layer
        """

        return True

