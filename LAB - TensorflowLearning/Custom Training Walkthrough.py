#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

ds_preview, info = tfds.load('penguins/simple', split='train', with_info=True)
df = tfds.as_dataframe(ds_preview.take(5), info)
print(df)
print(info.features)

class_names = ['Adélie', 'Chinstrap', 'Gentoo']

ds_split, info = tfds.load("penguins/processed", 
                           split=['train[:20%]', 'train[20%:]'], 
                           as_supervised=True, with_info=True)

ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)


df_test = tfds.as_dataframe(ds_test.take(5), info)


df_train = tfds.as_dataframe(ds_train.take(5), info)


ds_train_batch = ds_train.batch(32)

features, labels = next(iter(ds_train_batch))


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), #Input shape is required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

predictions = model(features)
print(predictions[:5])