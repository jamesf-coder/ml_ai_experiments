# https://www.tensorflow.org/tutorials


import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# train
def train_symbols():

    IMG_DIM = 128
    CAT_COUNT = 3
    TRAIN_COUNT = 80*CAT_COUNT
    TEST_COUNT = 20*CAT_COUNT

    def load_image(path: str):
        # return numpy array that has shape(128,128)
        # open the PNG and convert to luminance
        png = Image.open(os.path.join(SRC_DIR, path)).convert("L")
        # png.save("matplotlib_mono.png")

        # copy to numpy as a uint8
        png_array = np.array(png, dtype=np.uint8)
        
        # Reshape the array to include a channel dimension
        # image_array = np.reshape(image_array, (image_array.shape[0], image_array.shape[1], 1))
        
        # Normalize the pixel values to [0, 1]
        png_array = png_array / 255.0
        
        # Convert the array to float32 data type and flatten to 1D
        # png_array = png_array.astype('float32').flatten()
        png_array = png_array.astype('float32')

        png_array = png_array.squeeze()
        return png_array

    # load training data with shape(80,128,128)
    x_train:np.array[np.array] = np.empty((TRAIN_COUNT,IMG_DIM,IMG_DIM))
    y_train:np.array[np.array] = np.empty(TRAIN_COUNT)
    idx = 0
    for img in range(0,80):
        # images are 128x128
        x_train[idx] = load_image(f"symbols/rectangles/draw_rect_{str(img+1)}.png")
        y_train[idx] = 1
        idx = idx + 1
    for img in range(0,80):
        # images are 128x128
        x_train[idx] = load_image(f"symbols/circles/draw_circle_{str(img+1)}.png")
        y_train[idx] = 2
        idx = idx + 1
    for img in range(0,80):
        # images are 128x128
        x_train[idx] = load_image(f"symbols/triangles/draw_triangle_{str(img+1)}.png")
        y_train[idx] = 3
        idx = idx + 1

    # load testing data with shape(20,128,128)
    x_test:np.array[np.array] = np.empty((TEST_COUNT,IMG_DIM,IMG_DIM))
    y_test:np.array[np.array] = np.empty(TEST_COUNT)
    idx = 0
    for img in range(0,20):
        # images are 128x128
        x_test[idx] = load_image(f"symbols/rectangles/draw_rect_{str(img+51)}.png")
        y_test[idx] = 1
        idx = idx + 1
    for img in range(0,20):
        # images are 128x128
        x_test[idx] = load_image(f"symbols/circles/draw_circle_{str(img+51)}.png")
        y_test[idx] = 2
        idx = idx + 1
    for img in range(0,20):
        # images are 128x128
        x_test[idx] = load_image(f"symbols/triangles/draw_triangle_{str(img+51)}.png")
        y_test[idx] = 3
        idx = idx + 1

    #preprocessed_image = image.img_to_array(image_array)

    # remove me 
    # mnist = tf.keras.datasets.mnist

    # (xx_train, yy_train), (xx_test, yy_test) = mnist.load_data()
    # xx_train, xx_test = xx_train / 255.0, xx_test / 255.0
   

    # TODO: Check out keras core: https://keras.io/keras_core/announcement/

    # Sequential is useful for stacking layers where each layer 
    # has one input tensor and one output tensor. Layers are 
    # functions with a known mathematical structure that can be 
    # reused and have trainable variables. Most TensorFlow 
    # models are composed of layers. This model uses the 
    # Flatten, Dense, and Dropout layers.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMG_DIM, IMG_DIM)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.summary()

    model.compile(optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])    

    # training
    model.fit(x_train, y_train, epochs=5)

    # check performance
    model.evaluate(x_test,  y_test, verbose=2)

    # tell model to return a probability
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])


    # Steps (as shown at https://keras.io/keras_core/announcement/)
    #model.compile
    #model.fit
    #model.evaluate
    #model.predict

    predictions = model(x_train[:1]).numpy()


# test
def test_symbols():
    pass


train_symbols()
