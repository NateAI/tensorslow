
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

        self.prev_layer_output = None
        self.activations = None

    def forward_pass(self, prev_layer_output):
        """ σ(z_i) =  e^(z_i) / sum({e^(z_j) for all j}) """

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        self.prev_layer_output = prev_layer_output
        self.activations = np.exp(prev_layer_output) / np.sum(np.exp(prev_layer_output), axis=1)[:, None]

        return self.activations

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        """
        dσ(o_i)/d(o_j) = p_i (δ_i,j - p_j) where δ_i,j = 1 if i=j, 0 otherwise (i.e. Kronecker delta)
        Parameters
        ----------
        next_layer_gradients: [batch_size, num_output_neurons]

        Returns
        -------
        gradients: np.ndarray
            [batch_size, num_neurons]

        """

        num_neurons = next_layer_gradients.shape[1]
        batch_size = next_layer_gradients.shape[0]

        kronecker_array = np.repeat(np.eye(num_neurons)[None, :, :], batch_size, axis=0)
        activations_matrices = np.repeat(self.activations[:, None], num_neurons, axis=1)
        activations_matrices_transpose = np.swapaxes(activations_matrices, 1, 2)


        gradients_matrix = activations_matrices_transpose * (kronecker_array - activations_matrices)

        gradients = np.array([np.matmul(next_layer_gradients[:, None, :][i], gradients_matrix[i]) for i in range(batch_size)])
        gradients = np.squeeze(gradients, axis=1)  # get rid of extra dim

        return gradients




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