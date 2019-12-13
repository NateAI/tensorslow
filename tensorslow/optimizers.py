
import numpy as np


class Optimizer:

    """ Base class for all optimizers"""

    def get_updated_weights(self, current_weights, weight_gradients):
        """
        compute updated weights

        Parameters
        ----------
        current_weights: list[np.ndarray]
            current model weights
        weight_gradients: list[np.ndarray]
            list containing partial derivities for all weights - shapes should match the current_weights

        Returns
        -------
        updated_weights: list[np.ndarray]
        """
        raise NotImplementedError(
            'You must implement the update_weights method of any layer inheriting from ParametricLayer')

    def _verify_update_weights_input(self, current_weights, weight_gradients):

        class_name = self.__class__.__name__

        if not all([w1.shape == w2.shape for w1, w2 in zip(current_weights, weight_gradients)]):
            raise ValueError('Input parameters current_weights and weight_gradients to {} layer must have same shape'.format(class_name))


class SGD(Optimizer):

    def __init__(self, lr):
        """
        Stochastic Gradient Descent
        Parameters
        ----------
        lr:
        """
        self.lr = lr

    def get_updated_weights(self, current_weights, weight_gradients):

        self._verify_update_weights_input(current_weights, weight_gradients)

        updated_weights = []
        for current_weight_arr, weight_gradient_arr in zip(current_weights, weight_gradients):
            updated_weights.append(current_weight_arr - (self.lr * weight_gradient_arr))

        return updated_weights

