
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
        Compute partial derivatives of loss function wrt input to this layer (logits)

        Parameters
        ----------
        next_layer_gradients: [batch_size, num_output_neurons]

        Returns
        -------
        gradients: np.ndarray
            [batch_size, 1, num_neurons]

        """

        # Get partial derivatives of softmax activations wrt logits (Jacobian matrix)
        jacobian = self.softmax_gradients()

        gradients = np.matmul(next_layer_gradients, jacobian)  # chain rule to compute ∂L/∂z_i

        return gradients

    def softmax_gradients(self):

        """
        Compute partial derivatives of softmax activations (probabilities) wrt each logit

        ∂(p_i)/∂(z_j) = p_i(δ_i,j - p_j) where δ_i,j = 1 if i=j, 0 otherwise (i.e. Kronecker delta)

        where p_i = σ(z_i) is the softmax activation (probability) for logit z_i

        Returns
        -------
        jacobian: np.ndarray
            [batch_size, num_neurons, num_neurons] jacobian matrix
        """

        num_neurons = self.activations.shape[1]
        batch_size = self.activations.shape[0]

        p = np.repeat(self.activations[:, None], num_neurons, axis=1)
        p_transpose = np.swapaxes(p, 1, 2)
        kronecker = np.repeat(np.eye(num_neurons)[None, :], batch_size, axis=0)

        jacobian = p_transpose * (kronecker - p)

        return jacobian


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