#Switching CPU operation instructions to AVX AVX2
import os
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.gen_batch_ops import batch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Split the data manually into 80% training, 10% testing and 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
'cats_vs_dogs',
split=[
  tfds.Split.TRAIN.subsplit(tfds.percent[:80]),
  tfds.Split.TRAIN.subsplit(tfds.percent[80:90]),
  tfds.Split.TRAIN.subsplit(tfds.percent[90:])],
with_info=True,
as_supervised=True,
)


IMG_SIZE = 160 #All images will be resized to 160x160

def format_example(image,label):
  """
  Returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image,tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE,IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

get_label_name = metadata.features['label'].int2str #Creates a function object that we can use to get labels

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

##Prediction script
COLOR = 'blue'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, num):
  class_names = ['Cat','Dog']
  for img, lbl in test.take(num):
    label = lbl
  predicted_class = class_names[label]

  show_image(image, class_names[label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
for img, label in test.take(num):
  image = img
  predict(new_model, image, num)
#This project needs fixing