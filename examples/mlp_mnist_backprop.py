import itertools
import random

from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.datasets import mnist


from tensorslow.layers import FullyConnected, Sigmoid, Softmax, CategoricalCrossentropy
from tensorslow.models import Model
from tensorslow.optimizers import SGD

""" In this example we will train a MLP in Keras to classify handwritten digits in the MNIST dataset.

    We will then load the trained weights from the Keras model into the equivilent tensorslow implementation in order
    to check it's predictions work. 
 """

#######################################################################################################################
#                                 STEP 1 - Load and process MNIST dataset
#######################################################################################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255  # normalize pixel values to range [0, 1]
x_test = x_test / 255
print("Training data shape: ", x_train.shape)
print("Test data shape", x_test.shape)

# Display a grid of randomly chosen example images
rows = 5
columns = 5
fig, ax = plt.subplots(rows, columns)
for row, column in itertools.product(range(rows),
                                     range(columns)):  # loops through all permutations of row and column numbers
    random_img_idx = random.randint(0, len(x_train))  # get random img idx to plot
    ax[row, column].imshow(x_train[random_img_idx])
    ax[row, column].axis('off')
fig.suptitle('Example images from the MNIST dataset', fontsize=6)
plt.show()  # makes plot visible

# Flatten the images
image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
print('\n Images have been flatten to a {} dimensional vector:'.format(x_train[0].shape))

# Convert integer labels to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(' Training labels converted from integers to one-hot vectors with shape: {}'.format(y_train[0].shape))

#######################################################################################################################
#                                 STEP 2 - Implement MLP in tensorslow
#######################################################################################################################

tensorslow_model = Model()
tensorslow_model.add_layer(FullyConnected(neurons=100, input_dim=784))  # for now you have to specify input dim of every parametric layer
tensorslow_model.add_layer(layer=Sigmoid())
tensorslow_model.add_layer(layer=FullyConnected(neurons=10, input_dim=100))
tensorslow_model.add_layer(layer=Softmax())
sgd = SGD(lr=0.1)
tensorslow_model.compile(loss=CategoricalCrossentropy(), optimizer=sgd, metrics=['accuracy'])

# Test that the forward pass is working

# batch = x_train[:32]
# y_true = y_train[:32]
# y_pred = tensorslow_model.predict(batch)
# print('y_pred shape is: ', y_pred.shape)
#
# performance_dict = tensorslow_model.evaluate(batch, y_true)
# print('performance of untrained model is: \n', performance_dict)
#
# weight_gradients = tensorslow_model._backward_pass()
#
# updated_weights = tensorslow_model.optimizer.get_updated_weights(current_weights=tensorslow_model.weights,
#                                                                  weight_gradients=weight_gradients)
# tensorslow_model.set_weights(updated_weights)

logs = tensorslow_model.train(x_train[:5000], y_train[:5000], batch_size=32, epochs=15, validation_data=(x_test[:100], y_test[:100]))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(logs['loss']['train'])
ax1.plot(logs['loss']['val'])
ax1.set_title('Loss History')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(['Train', 'Val'])

ax2.plot(logs['accuracy']['train'])
ax2.plot(logs['accuracy']['val'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.set_title('Accuracy History')
ax2.legend(['Train', 'Val'])

fig.suptitle('Tensorslow MNIST MLP')
plt.show()


