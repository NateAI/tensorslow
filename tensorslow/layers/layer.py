
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





