import numpy as np

from tensorslow.layers.parametric_layers.parametric_layer import ParametricLayer


class FullyConnected(ParametricLayer):

    def __init__(self, neurons, input_dim=None, initializer_name='GlorotUniform', initializer=None):

        super(FullyConnected, self).__init__(neurons, input_dim, initializer_name, initializer)

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
            partial derivatives of loss wrt outputs of this layer [batch_size, num_output_neurons]

        args
        kwargs

        Returns
        -------
        gradients: np.ndarray
            partial derivatives of loss wrt inputs of this layer
        """

        jacobian = np.transpose(self.weights)  # [num_output_neurons, num_input_neurons]

        gradients = np.matmul(next_layer_gradients, jacobian)

        return gradients

    def get_weight_gradients(self, next_layer_gradients, *args, **kwargs):
        """
        Compute partial derivatives of the loss wrt the weights (not inc. bias)
        Parameters
        ----------
        next_layer_gradients: np.ndarray
            partial derivatives of loss wrt outputs of this layer [batch_size, 1, num_output_neurons]
        args
        kwargs

        Returns
        -------
        mean_gradients: np.ndarray
            the mean partial derivatives of loss wrt weights across all examples in the batch
             [num_input_neurons, num_output_neurons]
        """

        next_layer_gradients = np.expand_dims(next_layer_gradients, axis=1)  # [batch_size, 1, num_neurons]

        # Create jacobian assuming that weights matrix is flattened into a [1 * neurons * input_dim] with all the weights
        # for the first input first
        batch_size = next_layer_gradients.shape[0]
        jacobian = np.zeros(shape=(batch_size, self.neurons, self.neurons * self.input_dim))
        for batch_idx in range(batch_size):
            for idx in range(self.neurons):
                jacobian[batch_idx][idx, self.input_dim * idx: self.input_dim * (idx + 1)] = self.prev_layer_output[batch_idx]

        gradients = np.matmul(next_layer_gradients, jacobian)  # [batch_size, 1, input_dim * neurons]

        # Unflatten weight grads to [input_dim, neurons] from [1, input_dim * neurons]
        gradients = np.swapaxes(np.reshape(gradients, (batch_size, self.neurons, self.input_dim)), 1, 2)  # [batch_size, input_dim, neurons]

        mean_gradients = np.mean(gradients, axis=0)

        return mean_gradients

    def get_bias_gradients(self, next_layer_gradients, *args, **kwargs):
        """
        Compute partial derivatives of the loss wrt the biases
        Parameters
        ----------
        next_layer_gradients: np.ndarray
            partial derivatives of loss wrt outputs of this layer [batch_size, 1, num_output_neurons]
        args
        kwargs

        Returns
        -------
        mean_gradients: np.ndarray
            [1, neurons]
        """

        # Because the partial derivatives of the logits wrt the bias is always one - the partial derivaties wrt the bias
        # is simply equal to the next_layer_gradients. We then average over the batch dimension.
        mean_gradients = np.mean(next_layer_gradients, axis=0)[None, :]

        return mean_gradients
