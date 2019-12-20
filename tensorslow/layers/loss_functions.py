""" All loss functions inherit from the layer class"""

import numpy as np


class Loss:
    """ Base class for all loss functions - probably this should also inherit from Layer in the future"""

    def __init__(self):

        # store these for gradient calculation
        self.y_true = None
        self.y_pred = None

    def forward_pass(self, y_true, y_pred):
        raise NotImplementedError(
            'You must implement the forward_pass method of any layer inheriting from ParametricLayer')

    def backward_pass(self):
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
        """∑_classes(target_class_i * log(prob_i))) / num_samples

        Returns
        -------
        losses: np.ndarray
            [batch_size, 1]
        """

        self._verify_forward_pass_input(y_true, y_pred)
        self.y_true = y_true
        self.y_pred = y_pred

        losses = -1 * np.sum((y_true * np.log(y_pred + np.finfo(float).eps)), axis=1)

        losses = np.expand_dims(losses, axis=1)

        return losses

    def backward_pass(self):
        """
        dLi/d(pi,j) = - yi,j / pi,j where i is the example index (row) and j is the class index (column)

        Returns
        -------
        gradients: np.ndarray
            [batch_size, number_neurons_in_prev_layer]
        """
        gradients = - self.y_true / self.y_pred

        return gradients
