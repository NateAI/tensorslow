"""
Base class for all layers
"""
from abc import abstractmethod

import numpy as np


class Layer(object):
    """ Base class for all layers"""

    def forward_pass(self, prev_layer_output, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the forward_pass method of any layer inheriting from ParametricLayer')

    def backward_pass(self, next_layer_gradients, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the forward_pass method of any layer inheriting from ParametricLayer')

    def _verify_forward_and_backward_pass_input(self, input_array):
        """ Perform checks on input to foward_pass and backward_pass - should be called by all child classes """

        class_name = self.__class__.__name__

        if not isinstance(input_array, np.ndarray):
            raise ValueError(
                'Input to this method of the {} layer must be a numpy array - recieved input of type: {}'.format(
                    class_name, type(input_array)))

        if not np.ndim(input_array) > 1:
            raise ValueError('Input to this method of the {} layer must be an array of at least '
                             '2 dimensions - recieved input of shape: {}'.format(class_name, input_array.shape))


class ParametricLayer(Layer):
    """ Base class for all parametrised layers i.e. those with weights and bias"""

    def __init__(self, neurons, input_dim=None):
        """

        Parameters
        ----------
        neurons: int
            number of neurons (units) in this layer
        input_dim: int, optional
            number of neurons (units) in previous layer - optional but can only be set once
        """
        if not isinstance(neurons, int):
            raise ValueError('neurons parameter must be an int')
        elif input_dim is not None and not isinstance(input_dim, int):
            raise ValueError('input_dim parameter must be an int')

        self._neurons = neurons
        self._input_dim = input_dim

        if input_dim is not None:
            self._initialise_weights_and_bias()
        else:
            self._weights = None
            self._bias = None

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, value):
        raise Warning('Cannot change number of neurons in layer after instantiation')

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        if self._input_dim is None:
            self._input_dim = value
            self._initialise_weights_and_bias()
        else:
            raise Warning('Cannot change input_dim of layer after instantiation')

    @property
    def weights(self):
        if self._weights is None:
            raise Warning('weights matrix has not been initialised yet - input_dim must be set first')
        else:
            return self._weights

    @weights.setter
    def weights(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError('weights must be a numpy array')
        elif not value.shape == self._weights.shape:
            raise ValueError('weights must be of shape: {} not {}'.format(self._weights.shape, value.shape))
        else:
            self._weights = value

    @property
    def bias(self):
        if self._bias is None:
            raise Warning('bias vector has not been initialised yet - input_dim must be set first')
        else:
            return self._bias

    @bias.setter
    def bias(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError('weights must be a numpy array')
        elif not value.shape == self._bias.shape:
            raise ValueError('weights must be of shape {} not {} '.format(self._bias.shape, value.shape))
        else:
            self._bias = value

    def _initialise_weights_and_bias(self):
        """ randomly initialise weights"""
        self._weights = np.random.random(size=(self.input_dim, self.neurons))
        self._bias = np.zeros(shape=self.neurons)

    @abstractmethod
    def get_weight_gradients(self, next_layer_gradients, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the get_weight_gradients method of any layer inheriting from ParametricLayer')

    @abstractmethod
    def get_bias_gradients(self, next_layer_gradients, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the get_bias_gradients method of any layer inheriting from ParametricLayer')


