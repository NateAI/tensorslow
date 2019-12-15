import time

from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Softmax as K_Softmax, Activation
from keras.models import Sequential
from keras.optimizers import SGD as K_SGD

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

fc1_units = 100
lr = 0.01

tensorslow_model = Model()
tensorslow_model.add_layer(FullyConnected(neurons=fc1_units, input_dim=image_vector_size))  # for now you have to specify input dim of every parametric layer
tensorslow_model.add_layer(layer=Sigmoid())
tensorslow_model.add_layer(layer=FullyConnected(neurons=10, input_dim=fc1_units))
tensorslow_model.add_layer(layer=Softmax())
sgd = SGD(lr=lr)
tensorslow_model.compile(loss=CategoricalCrossentropy(), optimizer=sgd, metrics=['accuracy'])

#######################################################################################################################
#                                 STEP 2 - Implement Equivilent MLP in Keras
#######################################################################################################################
# Build equivalent model in Keras
k_model = Sequential()
k_model.add(Dense(units=fc1_units, input_dim=image_vector_size))
k_model.add(Activation('sigmoid'))
k_model.add(Dense(units=num_classes, input_dim=fc1_units))
k_model.add(K_Softmax())

k_optimizer = K_SGD(lr=lr)
k_model.compile(optimizer=k_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#######################################################################################################################
#                                 STEP 3 - Give TS model the same initialized weights as the Keras model
#######################################################################################################################

# Set Tensorslow model to have same initial weights as keras model
# k_init_weights = k_model.get_weights()
# k_init_weights[1] = k_init_weights[1][None, :]  # add batch dim to keras biases before setting
# k_init_weights[3] = k_init_weights[3][None, :]  #
# tensorslow_model.set_weights(weights=k_init_weights)

#######################################################################################################################
#                                 STEP 3 - Train each Model
#######################################################################################################################

num_examples_to_use = 1000
epochs = 5
batch_size = 32

print('\n Training Keras Model')
start = time.time()
history = k_model.fit(x_train[:num_examples_to_use], y_train[:num_examples_to_use], batch_size=batch_size, epochs=epochs, shuffle=False)
k_runtime = time.time() - start

print('\n Training Tensorslow Model')
start = time.time()
logs = tensorslow_model.train(x_train[:num_examples_to_use], y_train[:num_examples_to_use], batch_size=batch_size, epochs=epochs, shuffle=False)
t_runtime = time.time() - start

print('\n Keras Runtime: {}  Tensorslow Runtime:  {}'.format(k_runtime, t_runtime))
print('\n Keras trained {} times faster than tensorslow'.format(t_runtime / k_runtime))
#######################################################################################################################


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(logs['loss']['train'])
ax1.plot(history.history['loss'])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.legend(['Tensorslow', 'Keras'])

ax2.plot(logs['accuracy']['train'])
ax2.plot(history.history['acc'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(['Tensorslow', 'Keras'])

fig.suptitle('MNIST MLP: Keras vs Tensorslow')
plt.show()


