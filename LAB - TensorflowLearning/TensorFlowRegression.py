from contextlib import suppress
from datetime import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.isna().sum()
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = preprocessing.Normalization(axis=1)
normalizer.adapt(np.array(train_features))

#Putting all of the features (data columns) in an array
horsepower = np.array(train_features['Horsepower'])

#Configuring the normalizer so the values of the input is between 0 and 1
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

#Configuring the model layers
horsepower_model = tf.keras.Sequential([
  horsepower_normalizer,
  layers.Dense(units=1)
])

#Configuring the optimizer and loss function to use
horsepower_model.compile(
  optimizer=tf.optimizers.Adam(learning_rate=0.1),
  loss='mean_absolute_error'
)

#Training the model to predict MPG based on horsepower
history = horsepower_model.fit(
  train_features['Horsepower'], train_labels,
  epochs=100,
  #Supresses logging of the execution (No Massive wall of text?)
  verbose=0,
  #Calculate the validation results on 20% of the training data
  validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0,10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)



test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
  test_features['Horsepower'],
  test_labels, verbose=0)

x = tf.linspace(0.0,250,251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label ='Data')
  plt.plot(x,y,color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()

plot_horsepower(x,y)
plt.show()

linear_model = tf.keras.Sequential([
  normalizer,
  layers.Dense(units=1)
])

linear_model.compile(
  optimizer=tf.optimizers.Adam(learning_rate=0.1),
  loss="mean_absolute_error"
)


history = linear_model.fit(
  train_features, train_labels,
  epochs=100,
  # Supress logging
  verbose=0,
  validation_split = 0.2)

plot_loss(history)
plt.show()

test_results['linear_model'] = linear_model.evaluate(
  test_features,test_labels, verbose=0
)