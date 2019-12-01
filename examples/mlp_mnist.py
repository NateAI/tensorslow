import itertools
import random

from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from keras.optimizers import SGD
from keras.utils import plot_model

from tensorslow.layers import FullyConnected, Sigmoid, Softmax, CategoricalCrossentropy, Relu, Tanh
from tensorslow.models import Model

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
for row, column in itertools.product(range(rows), range(columns)):  # loops through all permutations of row and column numbers
  random_img_idx = random.randint(0, len(x_train))  # get random img idx to plot
  ax[row, column].imshow(x_train[random_img_idx])
  ax[row, column].axis('off')
fig.suptitle('Example images from the MNIST dataset', fontsize=6)
plt.show()  # makes plot visible

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
print('\n Images have been flatten to a {} dimensional vector:'.format(x_train[0].shape))

# Convert integer labels to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(' Training labels converted from integers to one-hot vectors with shape: {}'.format(y_train[0].shape))


#######################################################################################################################
#                                 STEP 2 - Implement and Train Keras MLP
#######################################################################################################################

# Build Model
keras_model = Sequential()
keras_model.add(Dense(units=100, input_shape=(784,)))
keras_model.add(Activation('sigmoid'))
keras_model.add(Dense(units=num_classes))
keras_model.add(Activation('softmax'))

print('\n Here is a summary of the keras model we just built... \n')
keras_model.summary()


# Train Model

epochs = 20
batch_size = 32
learning_rate = 0.01

optimizer = SGD(lr=learning_rate)

keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train on subset of training data so that we can observe the model improving over the epochs
history = keras_model.fit(x_train[:2000], y_train[:2000], batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, verbose=1)

# Plot training logs
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Val'])

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Val'])

plt.show()

#######################################################################################################################
#                                 STEP 3 - Implement Equivalent model in tensorslow
#######################################################################################################################


tensorslow_model = Model(loss=CategoricalCrossentropy())
tensorslow_model.add_layer(FullyConnected(neurons=100, input_dim=784))  # for now you have to specify input dim of every parametric layer
tensorslow_model.add_layer(layer=Sigmoid())
tensorslow_model.add_layer(layer=FullyConnected(neurons=10, input_dim=100))
tensorslow_model.add_layer(layer=Softmax())

# TODO add model summary method

#######################################################################################################################
#                        STEP 3 - Load trained weights into tensorslow model and test predictions
#######################################################################################################################

keras_weights = keras_model.get_weights()  # get trained keras weights
tensorslow_model.set_weights(keras_weights)  # set weights of our mlp to the trained weights
tensorslow_weights = tensorslow_model.get_weights()  # get updated mlp weights

# Check that weights have been set properly
for idx, (k_weight, ts_weight) in enumerate(zip(keras_weights, tensorslow_weights)):
    if (k_weight == ts_weight).all():
        print('Layer {} weights are the same'.format(idx))
    else:
        print('Warning! Layer {} weights are not the same for the keras and tensorslow models'.format(idx))


# Test that Tensorslow and Keras predictions match up

batch = x_test[:32]
keras_pred = keras_model.predict_on_batch(batch)
tensorslow_pred = tensorslow_model.predict(batch)

tolerance = 1e-4
if np.allclose(keras_pred, tensorslow_pred, atol=tolerance):
    print('Keras and Tensorslow predictions match within tolerance of: {}'.format(tolerance))
else:
    print('Keras and Tensorslow predictions do not match within tolerance of {}'.format(tolerance))


# Visualise tensorslow predictions
rows = 4
columns = 4
fig, ax = plt.subplots(rows, columns)
img_idx = 0
for row, column in itertools.product(range(rows), range(columns)):  # loops through all permutations of row and column numbers
    true_class_label = np.argmax(y_test[img_idx])
    pred_class_label = np.argmax(tensorslow_pred[img_idx])

    original_image = np.reshape(batch[img_idx], (28, 28))  # recover original 2D image from flattened vector
    ax[row, column].imshow(original_image)
    ax[row, column].set_title('True: {} - Pred: {}'.format(true_class_label, pred_class_label), fontsize=6)
    ax[row, column].axis('off')

    img_idx += 1  # move to next image in batch

plt.show()  # makes plot visible







