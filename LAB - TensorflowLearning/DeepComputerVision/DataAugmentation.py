

#Switching CPU operation instructions to AVX AVX2
import os
from sys import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Importing base tensorflow
import tensorflow as tf

#Importing tensorflow keras datasets, layers and models
from tensorflow.keras import datasets, layers, models
#Import pyplot so i can plot the results to a diagram
import matplotlib.pyplot as plt

#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)

##Dataloading and preprocessing
#Load and split dataset
(train_images,train_labels), (test_images,test_labels) = datasets.cifar10.load_data()
#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#Creates a data generator object that transforms images
datagen = ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

#Picking an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img) #Converting image to numpy array
img = img.reshape((1,) + img.shape) #Reshaping image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
  plt.figure(i)
  plot = plt.imshow(image.img_to_array(batch[0]))
  i += 1
  if i > 4:
    break

plt.show()