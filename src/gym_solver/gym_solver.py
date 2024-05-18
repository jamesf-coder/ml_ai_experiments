'''
Good place to start on Deep Reinforcement Learning:
* https://github.com/alessiodm/drl-zh
* https://github.com/alessiodm/drl-zh/blob/main/00_Intro.ipynb

'''

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras


print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\nKeras version = {keras.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')


