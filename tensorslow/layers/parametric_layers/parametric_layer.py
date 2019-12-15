from abc import abstractmethod
import inspect

import numpy as np

from tensorslow.layers.layer import Layer
from tensorslow.initializers import Initializer
import tensorslow.initializers


class ParametricLayer(Layer):
    """ Base class for all parametrised layers i.e. those with weights and bias"""

    def __init__(self, neurons, input_dim=None, initializer_name='GlorotUniform', initializer=None):
        """

        Parameters
        ----------
        neurons: int
            number of neurons (units) in this layer
        input_dim: int, optional
            number of neurons (units) in previous layer
            optional on instantiation because it can set later by observing the output dim of the previous layer
        initializer: str, optional
            name of a initializer e.g. 'GlorotUniform' - allows for easily defining initializer by name using default parameters
        initializer: tensorslow.initializers.Initializer, optional
            subclass of Initializer - allows for passing of initializer with custom parameters
        """
        # Checks on input parameters
        if not isinstance(neurons, int):
            raise ValueError('neurons parameter must be an int')
        elif input_dim is not None and not isinstance(input_dim, int):
            raise ValueError('input_dim parameter must be an int')
        elif initializer is not None and not isinstance(initializer, Initializer):
            raise ValueError('initializer parameter must be a sub-class of tensorslow.initializers.Initializer')
        elif not isinstance(initializer_name, str):
            raise ValueError('initializer parameter must be a string')

        # Note this is NOT necessarily the name of the initializer being used as it has a default value
        self._initializer_name = initializer_name
        self._initializer = initializer

        self._neurons = neurons
        self._input_dim = input_dim

        if input_dim is not None:
            self._set_initializer()
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
            self._set_initializer()  # must be set before weights can be initialized
            self._initialise_weights_and_bias()
        else:
            raise Warning('Cannot change input_dim of layer after it has been set')

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

    def _set_initializer(self):
        """
        Set the initializer object for the layer based on the input parameters to __init__
        """
        if self._initializer is not None:
            self.initializer = self._initializer
            self._initializer_name = self.initializer.__class__.__name__
        else:
            # Get dict mapping the name of available initializers to the class itself
            available_initializers_dict = {f[0]: f[1] for f in inspect.getmembers(tensorslow.initializers, inspect.isclass)}
            if self._initializer_name not in available_initializers_dict.keys():
                raise ValueError('initializer with name: {} was not found. \n Found the following options: {}'
                                 .format(self._initializer_name, list(available_initializers_dict)))
            else:
                self.initializer = available_initializers_dict[self._initializer_name](input_dim=self._input_dim, neurons=self._neurons)

    def _initialise_weights_and_bias(self):
        """Initialize weights and bias"""
        weights, bias = self.initializer.get_initial_weights_and_bias()
        self._weights = weights
        self._bias = bias

    @abstractmethod
    def get_weight_gradients(self, next_layer_gradients, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the get_weight_gradients method of any layer inheriting from ParametricLayer')

    @abstractmethod
    def get_bias_gradients(self, next_layer_gradients, *args, **kwargs):
        raise NotImplementedError(
            'You must implement the get_bias_gradients method of any layer inheriting from ParametricLayer')