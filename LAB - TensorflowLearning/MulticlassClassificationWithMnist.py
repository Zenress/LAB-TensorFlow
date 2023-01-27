#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#Standard imports ^

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('tkagg')

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

np.set_printoptions(linewidth=200)

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
print(x_train_normalized[2900][10])

def plot_curve(epochs, hist, list_of_metrics):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")
  
  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)
    
  plt.legend()
  plt.show()
  
def create_model(my_learning_rate):
  model = tf.keras.models.Sequential()
  
  #Flattening layer that flattens the 2 dimensional 28x28 array into a 784 element array
  model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
  #First hidden layer
  model.add(tf.keras.layers.Dense(units=256, activation='relu'))
  #Second hidden layer
  model.add(tf.keras.layers.Dense(units=128, activation='relu'))
  #Dropout regularization layer
  model.add(tf.keras.layers.Dropout(rate=0.1))
  #Output layer
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model

def train_model(model, train_featurers, train_label, epochs, batch_size=None, validation_split=0.1):
  history = model.fit(x=train_featurers, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True,
                      validation_split=validation_split)
  
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  
  return epochs, hist

#Hyperparemeters
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

my_model = create_model(learning_rate)

epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)

list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)