from numpy import dtype
#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import tensorflow as tf
from tensorflow import keras
import numpy as np

"""#Model number 1 - Basic ml model 
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Extended array (Helps visualise the end results). from xs to (-) ys
#5.0 - 9.0, 6.0 - 11.0, 7.0 - 13.0, 8.0 - 15.0, 9.0 - 17.0, 10.0 - 19.0
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
ys = np.array([-3.0, -1.0,1.0,3.0,5.0,7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))"""

"""#Model number 2 - Image recognition with uniform image sizes
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')

model.fit(train_images,train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)"""


#Model number 3 - Feature filters
"""model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64,(3,3), activation ='relu', input_shape=(28,28,1)),
  tf.keras.layers.MaxPoolingh2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
"""

#Final Model number 4
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
TRAINING_DIR = 'C:/Users/shan0382/Documents/GitHub/Job-CV-Coding/LAB -  Machine Learning Datasets/rpsTraining/rps/'
training_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
  TRAINING_DIR,
  target_size=(150,150),
  class_mode='categorical'
)

VALIDATION_DIR = 'C:/Users/shan0382/Documents/GitHub/Job-CV-Coding/LAB -  Machine Learning Datasets/rpsTesting/rps-test-set'
validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
  VALIDATION_DIR,
  target_size=(150,150),
  class_mode='categorical'
)

  #Model itself
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
    #Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
    #512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=25,
                validation_data = validation_generator,
                verbose=1)

model.save("rockpaperscissormodel.h5")