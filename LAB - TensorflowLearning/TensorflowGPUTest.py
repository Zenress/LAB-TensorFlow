import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())