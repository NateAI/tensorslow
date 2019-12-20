import os
import unittest

import numpy as np

from tensorslow.models import Model
from tensorslow.layers import FullyConnected, CategoricalCrossentropy
from tensorslow.layers import Softmax
from tensorslow.layers import Sigmoid
from tensorslow.optimizers import SGD


class TestModelCreation(unittest.TestCase):

    """ Test functions related to the creation of a model"""

    def setUp(self) -> None:
        self.model = Model()

    def test_add_first_layer_correcly(self):

        self.setUp()
        self.model.add_layer(FullyConnected(neurons=10, input_dim=20))

        # Test that the layer has been added to the layers attribute
        self.assertIsNotNone(self.model.layers, msg='Failed to add layer to layers attribute of model')

        # Test that the models weights attribute is not not empty
        model_weights_list = self.model.weights
        self.assertIsNotNone(model_weights_list, msg='Failed to add layers weights to the weights attribute of model')

        # Test that the weights list contains two np.ndarrays
        self.assertEqual(len(model_weights_list), 2, msg='Failed to layer weights to weights attribute correctly')

        # Note - testing of the weight shapes will be left to testing the initializer as this is not the responsibility
        # of the add_layer method

    def test_add_first_layer_without_input_dim(self):
        """ Test that ValueError exception is raised if you attempt to add the first parametric layer without
        the input_dim parameter specified
        """
        self.setUp()
        self.assertRaises(ValueError, self.model.add_layer, FullyConnected(neurons=10))

    def test_add_second_layer_with_wrong_input_dim(self):
        """Test that a ValueError exception is rasied if we add a second FC layer with an input dim that does not
        match the output shape of the last parametric layer"""
        self.setUp()
        self.model.add_layer(FullyConnected(neurons=10, input_dim=20))
        self.model.add_layer(Sigmoid())

        self.assertRaises(ValueError, self.model.add_layer, FullyConnected(neurons=5, input_dim=100))

    def test_add_second_layer_with_inferred_input_dim(self):
        """ Test that the input_dim of a second FC layer can be inferred correctly"""

        self.setUp()
        self.model.add_layer(FullyConnected(neurons=10, input_dim=20))
        self.model.add_layer(Sigmoid())

        # Add second FC layer without specifying input_dim
        self.model.add_layer(FullyConnected(neurons=5))

        # Test that the input_dim has been correctly inferred as the output_dim of the first FC layer
        self.assertEqual(self.model.layers[-1].input_dim, 10)



