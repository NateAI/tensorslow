import os
import unittest

import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Softmax as K_Softmax, Activation
from keras.models import Sequential
from keras.optimizers import SGD as K_SGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorslow.models import Model as T_Model
from tensorslow.layers import FullyConnected, CategoricalCrossentropy
from tensorslow.layers import Softmax as T_Softmax
from tensorslow.layers import Sigmoid as T_Sigmoid
from tensorslow.optimizers import SGD as T_SGD


class TestTrainStep(unittest.TestCase):

    """ Test a single train step of Tensorslow MLP classifier

    We will do this by creating the equivilent MLP in Keras, setting the initial weights of the two models to be equal
    and then performing one training step. We then check that the updated weights are equal.
    """

    def setUp(self) -> None:

        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = x_train / 255  # normalize pixel values to range [0, 1]

        # Flatten the images
        input_dim = 28 * 28
        self.x_train = x_train.reshape(x_train.shape[0], input_dim)

        # Convert integer labels to "one-hot" vectors using the to_categorical function
        num_classes = 10
        self.y_train = keras.utils.to_categorical(y_train, num_classes)

        fc1_units = 100

        # Build simple Tensorslow MLP
        t_model = T_Model()
        t_model.add_layer(FullyConnected(neurons=fc1_units, input_dim=input_dim))
        t_model.add_layer(T_Sigmoid())
        t_model.add_layer(FullyConnected(neurons=num_classes, input_dim=fc1_units))
        t_model.add_layer(T_Softmax())

        t_optimizer = T_SGD(lr=0.01)
        t_model.compile(loss=CategoricalCrossentropy(), optimizer=t_optimizer)

        # Build equivalent model in Keras
        k_model = Sequential()
        k_model.add(Dense(units=fc1_units, input_dim=input_dim))
        k_model.add(Activation('sigmoid'))
        k_model.add(Dense(units=num_classes, input_dim=fc1_units))
        k_model.add(K_Softmax())

        k_optimizer = K_SGD(lr=0.01)
        k_model.compile(optimizer=k_optimizer, loss='categorical_crossentropy')

        # Set Tensorslow model to have same initial weights as keras model
        k_init_weights = k_model.get_weights()
        k_init_weights[1] = k_init_weights[1][None, :]  # add batch dim to keras biases before setting
        k_init_weights[3] = k_init_weights[3][None, :]  #
        t_model.set_weights(weights=k_init_weights)

        self.t_model = t_model
        self.k_model = k_model

    def test_train_step(self):

        """ Perform a single step of training and compare the updated weights of the tensorslow and keras models
        to ensure they are the same"""

        x_batch = self.x_train[:32]
        y_batch = self.y_train[:32]

        # Train both models on same batch
        self.k_model.train_on_batch(x_batch, y_batch)
        self.t_model.train_on_batch(x_batch, y_batch)  #

        # Get updated weights for each model
        k_new_weights = self.k_model.get_weights()
        k_new_weights[1] = k_new_weights[1][None, :]  # add batch dim to keras bias before comparing
        k_new_weights[3] = k_new_weights[3][None, :]
        t_new_weights = self.t_model.weights

        # Compare updated weights
        self.assertTrue(all([np.allclose(k_new_weights[i], t_new_weights[i], atol=1e-5) for i in range(4)]))


if __name__ == '__main__':
    unittest.main()




