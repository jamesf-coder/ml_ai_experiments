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

    def load_image(path: str):
        image = Image.open(os.path.join(SRC_DIR, "symbols/rectangles/draw_rect_1.png")).convert("RGB")
        image_array = np.array(image)

        # Display the image using matplotlib
        # plt.imshow(image_array)
        # plt.axis('off')
        # plt.show()
        # time.sleep(100000)

        # Reshape the array to include a channel dimension
        image_array = np.reshape(image_array, (image_array.shape[0], image_array.shape[1], 1))
        # Normalize the pixel values to [0, 1]
        image_array = image_array / 255.0
        # Convert the array to float32 data type
        image_array = image_array.astype('float32')

        return image_array

    # load training data
    x_train:list[np.array] = []
    for img in range(1,80):
        # images are 128x128
        x_train.append(load_image(f"symbols/rectangles/draw_rect_{str(img)}.png"))

    # load testing data
    x_test:list[np.array] = []
    for img in range(80,101):
        # images are 128x128
        x_test.append(load_image(f"symbols/rectangles/draw_rect_{str(img)}.png"))

    #preprocessed_image = image.img_to_array(image_array)
    


    # Sequential is useful for stacking layers where each layer 
    # has one input tensor and one output tensor. Layers are 
    # functions with a known mathematical structure that can be 
    # reused and have trainable variables. Most TensorFlow 
    # models are composed of layers. This model uses the 
    # Flatten, Dense, and Dropout layers.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(128, 128)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()


# test
def test_symbols():
    pass


train_symbols()
