
import numpy as np

from tensorslow.layers.layer import Layer


class Activation(Layer):

    """Base class for all activations"""

    def __init__(self, input_dim=None):

        self._input_dim = input_dim
        self._neurons = input_dim  # input and output shape are equal for all activations

    # TODO there is some duplication here with code in paramteric_layer
    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, value):
        raise Warning('Cannot change number of neurons in layer after instantiation')

    @property
    def input_dim (self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        if self._input_dim is None:
            if isinstance(value, int):
                self._input_dim = value
                self._neurons = value
            else:
                raise value('input_dim must be of type int - you passed a value of type {}'.format(type(value)))
        else:
            raise Warning('Cannot change input_dim of layer after it has been set')


class Sigmoid(Activation):

    def __init__(self, input_dim=None):

        super(Sigmoid, self).__init__(input_dim=input_dim)

        self.prev_layer_output = None
        self.activations = None

    def forward_pass(self, prev_layer_output):
        """ σ(z) = 1 / (1 + e^(-z))"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        self.prev_layer_output = prev_layer_output
        self.activations = 1 / (1 + np.exp(-1 * prev_layer_output))

        return self.activations

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        """
        Compute partial derivatives of loss function wrt input to this layer (logits)

        Parameters
        ----------
        next_layer_gradients: [batch_size, num_neurons]

        Returns
        -------
        gradients: np.ndarray
            [batch_size, num_neurons]

        """

        jacobian = self.sigmoid_gradients()  # [batch_size, num_neurons]

        assert jacobian.shape == next_layer_gradients.shape
        gradients = next_layer_gradients * jacobian  # [batch_size, num_neurons]

        return gradients

    def sigmoid_gradients(self):
        """
        ompute partial derivatives of sigmoid activations (probabilities) wrt each logit
        Returns
        -------
        dσ(z)/d(z) = σ(z)(1 - σ(z))
        """
        return self.activations * (1 - self.activations)


class Softmax(Activation):

    def __init__(self, input_dim=None):

        super(Softmax, self).__init__(input_dim)

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

        next_layer_gradients = np.expand_dims(next_layer_gradients, axis=1)  #  [batch_size, 1, num_neurons]

        # Get partial derivatives of softmax activations wrt logits (Jacobian matrix)
        jacobian = self.softmax_gradients()

        gradients = np.matmul(next_layer_gradients, jacobian)  # chain rule to compute ∂L/∂z_i

        gradients = np.squeeze(gradients)

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


class Relu(Activation):

    def __init__(self, input_dim=None):

        super(Relu, self).__init__(input_dim)

        self.prev_layer_output = None
        self.activations = None

    def forward_pass(self, prev_layer_output):
        """ σ(z) = max(0, z)"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        self.prev_layer_output = prev_layer_output
        self.activations = np.maximum(prev_layer_output, np.zeros(shape=prev_layer_output.shape))

        return self.activations

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        """compute partial derivatives of loss wrt inputs to this layer (logits)

        Parameters
        ----------
        next_layer_gradients: np.ndarray
            [batch_size, num_neurons]

        Returns
        -------
        gradients: np.ndarray
            [batch_size, num_neurons]
        """

        jacobian = self.relu_gradients()

        gradients = next_layer_gradients * jacobian  # [batch_size, num_neurons]

        return gradients

    def relu_gradients(self):
        """ compute partial derivatives of relu output activations wrt logits

        σ'(z) = {1 if σ(z)>0, 0 otherwise}
        """

        jacobian = (self.activations > 0).astype(int)

        return jacobian


class Tanh(Activation):

    def __init__(self, input_dim=None):

        super(Tanh, self).__init__(input_dim)

        self.prev_layer_output = None
        self.activations = None

    def forward_pass(self, prev_layer_output):
        """ σ(z) = (e^z - e^-z) / (e^z + e^-z)"""

        self._verify_forward_and_backward_pass_input(prev_layer_output)

        self.prev_layer_output = prev_layer_output

        e_z = np.exp(prev_layer_output)
        e_minus_z = np.exp(-1 * prev_layer_output)

        self.activations = (e_z - e_minus_z) / (e_z + e_minus_z)

        return self.activations

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        """compute partial derivatives of loss wrt inputs to this layer (logits)

        Parameters
        ----------
        next_layer_gradients: np.ndarray
            [batch_size, num_neurons]

        Returns
        -------
        gradients: np.ndarray
            [batch_size, num_neurons]
        """

        jacobian = self.tanh_gradients()

        gradients = next_layer_gradients * jacobian

        return gradients

    def tanh_gradients(self):
        """compute partial derivatives of tanh output (activations) wrt input (logits_

        σ'(z) = 1 = (σ(z))^2
        """

        jacobian = 1 - self.activations * self.activations

        return jacobian
