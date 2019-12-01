"""
This file will contain the Model class that encapsulates the functionality of  a Deep Learning model.
"""

import numpy as np


class Model:

    def __init__(self, loss):
        self.layers = []  # empty list to store layer objects
        self.weights = []  # to store weights and bias for each layer
        self.loss = loss

    def _forward_pass(self, batch):
        """ get a list containing the output of every layer in the model for a given batch

        Parameters
        ---------
        batch: np.ndarray

        Returns
        --------
        activations: list[np.ndarray]
            list containing output from each layer
        """
        activations = []
        prev_layer_output = batch
        for idx, layer in enumerate(self.layers):
            current_layer_output = layer.forward_pass(prev_layer_output)
            activations.append(current_layer_output)
            prev_layer_output = current_layer_output

        return activations

    def predict(self, batch):
        """ method to perform predictions - returns activations of final layer"""

        activations = self._forward_pass(batch)
        return activations[-1]

    def compute_loss(self, true_class_labels, batch):
        """return mean loss on batch"""
        predicted_probs = self.predict(batch)
        loss_vector = self.loss.forward_pass(true_class_labels, predicted_probs)
        return loss_vector

    def _backward_pass(self):
        """ method to perform backward pass of model - i.e. backgropogation"""
        print('coming soon...')

    def add_layer(self, layer):
        """ method to add a layer to the model - mimics keras model.add()"""
        # TODO add more checking on the layer input and output shape
        self.layers.append(layer)

        if hasattr(layer, 'weights'):
            self.weights.append(layer.weights)
        if hasattr(layer, 'bias'):
            self.weights.append(layer.bias)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        """ method to set the weights - i.e. if we were loading saved weights"""
        if type(weights) != list:
            raise ValueError('weights must be a list of numpy arrays')
        elif len(weights) != len(self.weights):
            raise ValueError('weights must be a list of length: ', len(self.weights))
        elif not all([w1.shape == w2.shape for w1, w2 in zip(weights, self.weights)]):
            raise ValueError('Weights must be the same shape as the model weights')
        else:
            self.weights = weights
            weight_idx = 0
            for layer in self.layers:
                if hasattr(layer, 'weights') and np.ndim(weights[weight_idx]) == 2:
                    layer.weights = weights[weight_idx]
                    weight_idx += 1
                if hasattr(layer, 'bias') and np.ndim(weights[weight_idx]) == 1:
                    layer.bias = weights[weight_idx]
                    weight_idx += 1

            if weight_idx != len(self.weights):
                raise ValueError('Unexpectd error setting weights')