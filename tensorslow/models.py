"""
This file will contain the Model class that encapsulates the functionality of  a Deep Learning model.
"""
import inspect
import numpy as np

import tensorslow
import tensorslow.metrics as tensorslow_metrics


class Model:

    def __init__(self):
        self.layers = []  # empty list to store layer objects
        self.weights = []  # to store weights and bias for each layer
        self.loss = None
        self.optimizer = None
        self.metrics = None
        self.metrics_dict = None  # mapping from metric name to function

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

    def evaluate(self, batch, y_true):
        """
        evaluate model on batch of examples
        Parameters
        ----------
        batch: np.ndarray
        y_true: np.ndarray
            one-hot labels

        Returns
        -------
        performance_dict: dict
            dict mapping metric name to score
        """

        performance_dict = {}

        if self.loss is None:
            raise AttributeError('loss attribute not set - you must compile the model before evaluation')
        else:
            y_pred = self.predict(batch)
            losses = self.loss.forward_pass(y_true, y_pred)
            performance_dict['loss'] = np.mean(losses)

        if self.metrics:
            for metric in self.metrics:
                performance_dict[metric] = self.metrics_dict[metric](y_true, y_pred)

        return performance_dict

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

    def compile(self, loss, optimizer, metrics=None):
        """
        Compile model so that it can be trained or evaluated on
        Parameters
        ----------
        loss: tensorslow.layers.loss_functions.Loss
        optimizer:
        metrics: list[str]
            list of metrics to compute - e.g. 'accuracy'

        Returns
        -------

        """

        if isinstance(loss, tensorslow.layers.loss_functions.Loss):
            self.loss = loss
        else:
            raise ValueError('loss must be a sub-class of tensorslow.layers.loss_functions.Loss')

        # dict mapping metric name to function
        available_metrics_dict = {f[0]: f[1] for f in inspect.getmembers(tensorslow_metrics, inspect.isfunction)}
        available_metrics = list(available_metrics_dict)

        if not isinstance(metrics, list):
            raise ValueError('metrics must be a list of metric names e.g. accuracy')
        elif not set(metrics).issubset(available_metrics):
            invalid_metrics = list(set(metrics) - set(available_metrics))
            raise ValueError('following metrics are not implemented in the metrics.py file: {}'.format(invalid_metrics))
        else:
            self.metrics = metrics
            self.metrics_dict = {metric: metric_func for metric, metric_func in available_metrics_dict.items() if metric in metrics}

        # TODO same for optimizer

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