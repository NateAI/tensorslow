from abc import abstractmethod

import numpy as np


class Initializer:

    """Base class for all initializers"""

    def __init__(self, neurons, input_dim, *args, **kwargs):

        """

        Parameters
        ----------
        neurons: int
            number of neurons in this layer (the output dim)
        input_dim: int
            number of neurons in the previous layer (the input dim)
        args
        kwargs
        """

        self.neurons = neurons
        self.input_dim = input_dim

    @abstractmethod
    def get_initial_weights_and_bias(self):
        raise NotImplementedError(
            'You must implement the get_initial_weights_and_bias method of any class inheriting from Initializer')


class RandomNormal(Initializer):

    def __init__(self, neurons, input_dim, mean, stddev):

        """
        weights are sampled from a normal N~(mean, stddev) distribution

        bias is all zeros by default


        Parameters
        ----------
        mean: float
            mean of the normal distribution
        stddev: float
            standard deviation for the normal distribution
        """

        self.mean = mean
        self.stddev = stddev
        super(RandomNormal, self).__init__(input_dim, neurons)

    def get_initial_weights_and_bias(self):

        weights = np.random.normal(loc=self.mean, scale=self.stddev, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias


class RandomUniform(Initializer):

    def __init__(self, neurons, input_dim, min=-0.1, max=0.1):

        """
        Parameters
        ----------
        min: float
            the lower limit of the uniform distribution
        max: float
            the upper limit of the uniform distribution
        """

        self.min = min
        self.max = max
        super(RandomUniform, self).__init__(input_dim, neurons)

    def get_initial_weights_and_bias(self):
        """
        weights are sampled from a uniform U~(min, max) distribution.

        bias is all zeros by default

        Returns
        -------
        weights: np.ndarray
            [input_dim, neurons]
        bias: np.ndarray
            [1, neurons]
        """

        weights = np.random.uniform(low=self.min, high=self.max, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias


class GlorotNormal(Initializer):

    def get_initial_weights_and_bias(self):
        """
        Weights are sampled from normal distribution N~(μ=0, σ=sqrt(2/(input_dim + neurons))

        bias is all zeros by default

        Source:

            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf - Equation (12)

        This initializers was designed for use with tanh or sigmoid activation functions

        TODO: Implement truncated normal version

        Returns
        -------
        weights: np.ndarray
            [input_dim, neurons]
        bias: np.ndarray
            [1, neurons]
        """

        stddev = np.math.sqrt(2 / (self.input_dim + self.neurons))
        weights = np.random.normal(loc=0, scale=stddev, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias


class GlorotUniform(Initializer):

    def get_initial_weights_and_bias(self):
        """
        Weights are sampled from a uniform U~(-lim, lim) distribution where lim=sqrt(6 / (input_dim + neurons))

        bias is all zeros by default

        source:

        https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404

        (cannot find original paper)

        Returns
        -------
         weights: np.ndarray
            [input_dim, neurons]
        bias: np.ndarray
            [1, neurons]
        """

        lim = np.math.sqrt(6 / (self.input_dim + self.neurons))
        weights = np.random.uniform(low=-lim, high=lim, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias


class HeNormal(Initializer):

    def get_initial_weights_and_bias(self):

        """
        Weights are sampled from a normal distribution N~(μ=0, σ=sqrt(2/(input_dim))

        bias is all zeros by default

        source:

            https://arxiv.org/pdf/1502.01852.pdf  Equation (10)

        This initializer was designed for use with ReLu activation function.

        TODO: Implement truncated normal version

        Returns
        -------
        weights: np.ndarray
            [input_dim, neurons]
        bias: np.ndarray
            [1, neurons]
        """

        stddev = np.math.sqrt(2 / self.input_dim)
        weights = np.random.normal(loc=0, scale=stddev, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias


class HeUniform(Initializer):

    def get_initial_weights_and_bias(self):

        """
        Weights are sampled from a uniform U~(-lim, lim) distribution where lim=sqrt(6 / (input_dim))

        bias is all zeros by default

        Returns
        -------
        weights: np.ndarray
            [input_dim, neurons]
        bias: np.ndarray
            [1, neurons]
        """

        lim = np.math.sqrt(6 / (self.input_dim ))
        weights = np.random.uniform(low=-lim, high=lim, size=(self.input_dim, self.neurons))
        bias = np.zeros(shape=(1, self.neurons))

        return weights, bias