'''
For classifiction, we can use a single output neuron - which outputs in a range 0..1 (ie. the estimated probability)
'''

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

# constants
XDIM = 28
YDIM = 28
VAL_NUM = 5000
EPOCHS = 30

fashion_mnist = keras.datasets.fashion_mnist 
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# X = pixel data
# y = classification number (see class_names below)

# training and test data already exist - but there's no validation so we have to create one
# also have to scale the pixels into the range 0..1
# NOTE: the validation dataset is optional
x_valid = X_train_full[:VAL_NUM] / 255.0
X_train = X_train_full[VAL_NUM:] / 255.0

y_valid = y_train_full[:VAL_NUM]
y_train = y_train_full[VAL_NUM:]
X_test = X_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[XDIM, YDIM])) # converts each input into a 1D array.  Could also use keras.layers.InputLayer
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) # outputlayer - 10 neurons for each classification type.  Softmax because classes are exclusive

# sgd = Stochastic Gradient Descent (ie use back-propagation)
optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train! - remember the validation parameter is optional.  Can use validation_split=0.1 instead
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(x_valid, y_valid))

# evaluate the model ON THE TEST DATA
print("Evaluation:")
print(model.evaluate(X_test, y_test))

