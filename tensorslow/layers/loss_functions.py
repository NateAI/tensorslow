""" All loss functions inherit from the layer class"""

import numpy as np


class Loss:
    """ Base class for all loss functions"""

    def forward_pass(self, y_true, y_pred):
        raise NotImplementedError(
            'You must implement the forward_pass method of any layer inheriting from ParametricLayer')

    def backward_pass(self, loss):
        raise NotImplementedError(
            'You must implement the forward_pass method of any layer inheriting from ParametricLayer')

    def _verify_forward_pass_input(self, y_true, y_pred):

        class_name = self.__class__.__name__

        if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
            raise ValueError('Input parameters y_true and y_pred to {} layer must be numpy arrays'.format(class_name))

        if not y_true.shape == y_pred.shape:
            raise ValueError('Input parameters y_true and y_pred to {} layer must have same shape'.format(class_name))


class CategoricalCrossentropy(Loss):

    def forward_pass(self, y_true, y_pred):
        """(∑_samples(∑_classes(target_class_i * log(prob_i))) / num_samples """

        self._verify_forward_pass_input(y_true, y_pred)

        return y_true * np.log(y_pred + np.finfo(float).eps)



