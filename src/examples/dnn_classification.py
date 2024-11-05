'''
For classifiction, we can use a single output neuron - which outputs in a range 0..1 (ie. the estimated probability)
'''

import os

import PIL.ImageOps
import PIL.ImageShow
os.environ["KERAS_BACKEND"] = "tensorflow"

import PIL
import tensorflow as tf
import keras
import numpy as np

# constants
MODEL_FNAME = "dnn_classification_example.keras"
XDIM = 28
YDIM = 28
VAL_NUM = 5000
EPOCHS = 30
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def train():

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
    model.save(MODEL_FNAME)

    print("Fit history:")
    print(history)

    # evaluate the model ON THE TEST DATA
    print("Evaluation:")
    print(model.evaluate(X_test, y_test))

print("================================================")

model = None
# test to see if we need to train the model (ie is the file present)
if not os.path.isfile(MODEL_FNAME):
    # train the model
    model = train()
else:
    print(f"Using cached model file at: {os.path.abspath(MODEL_FNAME)}")
    model = keras.models.load_model(MODEL_FNAME)

# make predications on some data never seen before
img = keras.preprocessing.image.load_img("src/examples/test_data/bag1.png", target_size=(XDIM, YDIM), color_mode="grayscale")
img = PIL.ImageOps.invert(img)
PIL.ImageShow.show(img)
input_arr = keras.utils.img_to_array(img)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(predictions)
for c in range(len(CLASS_NAMES)):
    print(f"{CLASS_NAMES[c]}: {predictions[0][c]}")

print("Fin")
