"""
This file will contain the Model class that encapsulates the functionality of  a Deep Learning model.
"""
import inspect
import numpy as np
from tqdm import tqdm

from tensorslow.layers import ParametricLayer, Loss
from tensorslow.optimizers import Optimizer
import tensorslow.metrics as tensorslow_metrics


class Model:

    def __init__(self):

        self.batch = None  # store each batch for the backward pass
        self.y_pred = None

        self.layers = []  # empty list to store layer objects
        self.weights = []  # to store weights and bias for each layer
        self.loss = None
        self.optimizer = None
        self.metrics = None  # list of metric names
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
        self.batch = batch
        activations = []
        prev_layer_output = batch
        for layer in self.layers:
            current_layer_output = layer.forward_pass(prev_layer_output)
            activations.append(current_layer_output)
            prev_layer_output = current_layer_output

        self.y_pred = activations[-1]

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
        """
        Performs single iteration of backward pass through full model
        Parameters
        ----------

        Returns
        -------
        weight_gradients: list[np.ndarray]
            list containing the update gradients for all weights and biases in the model
            the shape of all arrays in the list should match the shapes in self.weights
            the gradients are the averaged over the batch dimension for mini-batch learning
        """

        weight_gradients = []  # list to store gradients for all weights in model

        loss_grads = self.loss.backward_pass()
        next_layer_grads = loss_grads

        for layer in reversed(self.layers):
            current_layer_grads = layer.backward_pass(next_layer_grads)

            # If the layer is parametric (i.e. has weights and bias) the also compute the partial derivatives of the
            # weights and bias and store in output list. bias first to keep ordering of model weights.
            if isinstance(layer, ParametricLayer):
                weight_gradients.insert(0, layer.get_bias_gradients(next_layer_grads))
                weight_gradients.insert(0, layer.get_weight_gradients(next_layer_grads))

            # current_layer_grads become the next_layer_grads for the next iteration
            next_layer_grads = current_layer_grads

        if not all([w1.shape == w2.shape for w1, w2 in zip(weight_gradients, self.weights)]):
            raise Exception('Error: weight gradients do not have the same shape as the model weights')

        return weight_gradients

    def train(self, x_train, y_train, epochs, batch_size, validation_data=None, shuffle=True):
        """
        Train model
        Parameters
        ----------
        x_train: np.ndarray
            [num_examples, input_dim]
        y_train: np.ndarray
            [num_examples, target_dim]
        epochs: int
        batch_size: int
        validation_data: tuple, optional
            (x_val, y_val)
        shuffle: bool, optional

        Returns
        -------
        logs: dict
            {'loss': {'train': [1.32, 1.22, 1.02..]
                      'val': []},
             'accuracy': {'train': []
                          'val': []}
            dict containing list of all metric scores over time
        """

        #  TODO move these check somewhere else and improve them
        assert np.ndim(x_train) == 2
        assert np.ndim(y_train) == 2
        assert x_train.shape[0] == y_train.shape[0]  # check number of examples is same
        assert y_train.shape[1] == self.weights[-1].shape[1]  # check target shape is same as model output
        assert x_train.shape[1] == self.weights[0].shape[0]  # check input dim is same as that of first layer

        if validation_data is not None:
            assert isinstance(validation_data, tuple)
            assert len(validation_data) == 2
            x_val = validation_data[0]
            y_val = validation_data[1]
            assert x_val.shape[0] == y_val.shape[0]
            assert x_train.shape[1] == x_val.shape[1]
            assert y_train.shape[1] == y_val.shape[1]

        num_examples = x_train.shape[0]
        print('\n Training on {} Examples \n'.format(num_examples))

        logs = dict.fromkeys(['loss'] + self.metrics)
        for key in logs.keys():
            logs[key] = {'train': [], 'val': []}

        for epoch in range(epochs):

            if shuffle:
                perm = np.random.permutation(num_examples)
                x_train = x_train[perm]
                y_train = y_train[perm]

            num_batches = np.math.ceil(num_examples / batch_size)
            remainder = num_examples % batch_size  # final batch can be of smaller size

            # TODO create batch generator for this
            for batch_idx in tqdm(range(num_batches)):

                if batch_idx == (num_batches - 1) and remainder > 0:
                    x_batch = x_train[batch_size * batch_idx:]  # if last batch and batch_size does not divide num_examples
                    y_batch = y_train[batch_size * batch_idx:]  # then use all remaining examples in final batch
                else:
                    x_batch = x_train[batch_size * batch_idx: batch_size * (batch_idx + 1)]  # default case
                    y_batch = y_train[batch_size * batch_idx: batch_size * (batch_idx + 1)]

                # Train step
                self.train_step(x_batch, y_batch)

            # End of Epoch
            performance_dict = self.evaluate(x_train, y_train)
            for key in performance_dict.keys():
                logs[key]['train'].append(performance_dict[key])

            if validation_data is not None:
                # TODO this should be done in batches as well - for now just use small val sets
                x_val = validation_data[0]
                y_val = validation_data[1]
                performance_dict = self.evaluate(x_val, y_val)
                print('\n Epoch: {:04d}   val_los: {:.3f}   val_acc: {:.3f}'.format(epoch+1, performance_dict['loss'], performance_dict['accuracy']))
                for key in performance_dict.keys():
                    logs[key]['val'].append(performance_dict[key])

        return logs

    def train_step(self, x_batch, y_batch):
        """
        Perform single step of training on a batch of examples
        Includes forward pass, backward pass and weight updates
        Parameters
        ----------
        x_batch: np.ndarray
            [batch_size, input_dim]
        y_batch: np.ndarray
            [batch_size, target_dim]

        Returns
        -------
        performance_dict: dict
            {'loss': 1.33..,
             'accuracy': '0.82..'
        """
        # Forward pass
        performance_dict = self.evaluate(batch=x_batch, y_true=y_batch)

        # Compute weight gradients
        weight_gradients = self._backward_pass()

        # Compute updated weights
        updated_weights = self.optimizer.get_updated_weights(current_weights=self.weights,
                                                             weight_gradients=weight_gradients)

        # Set updated weights
        self.set_weights(updated_weights)

        return performance_dict, weight_gradients

    def add_layer(self, layer):
        """ method to add a layer to the model - mimics keras model.add()"""
        # TODO add more checking on the layer input and output shape

        existing_parametric_layers = [layer for layer in self.layers if isinstance(layer, ParametricLayer)]

        if isinstance(layer, ParametricLayer):
            # If input_dim of a parametric layer is unspecified infer it from the output of the previous layer
            if layer.input_dim is None:
                if not existing_parametric_layers:
                    raise ValueError('You must specify the input_dim for the first parametric layer in a model')
                else:
                    layer.input_dim = existing_parametric_layers[-1].neurons
            elif existing_parametric_layers:
                # If input_dim is specified then check that it matches the output shape of the previous layer
                if layer.input_dim != existing_parametric_layers[-1].neurons:
                    raise ValueError('input_dim: {} does not match the output shape of the previous parametric layer: {}'
                                     .format(layer.input_dim, existing_parametric_layers[-1].neurons))
                # TODO Warning! This will probably need to change to checking the output shape of the previous layer -
                # TODO  regardless of whether it was parametric as some nonparametric layers i.e. maxpooling will change
                # TODO   the shape

        self.layers.append(layer)
        if isinstance(layer, ParametricLayer):
            self.weights.append(layer.weights)
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

        if isinstance(loss, Loss):
            self.loss = loss
        else:
            raise ValueError('loss must be a sub-class of tensorslow.layers.loss_functions.Loss')

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError('optimizer must be a sub-class of tensorslow.optimizers.Optimizer')

        # dict mapping metric name to function
        available_metrics_dict = {f[0]: f[1] for f in inspect.getmembers(tensorslow_metrics, inspect.isfunction)}
        available_metrics = list(available_metrics_dict)

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError('metrics must be a list of metric names e.g. [\'accuracy\']')
            elif not set(metrics).issubset(available_metrics):
                invalid_metrics = list(set(metrics) - set(available_metrics))
                raise ValueError('following metrics are not implemented in the metrics.py file: {}'.format(invalid_metrics))
            else:
                self.metrics = metrics
                self.metrics_dict = {metric: metric_func for metric, metric_func in available_metrics_dict.items() if metric in metrics}

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
                if isinstance(layer, ParametricLayer):
                    layer.weights = weights[weight_idx]
                    layer.bias = weights[weight_idx + 1]
                    weight_idx += 2

            if weight_idx != len(self.weights):
                raise ValueError('Unexpectd error setting weights')