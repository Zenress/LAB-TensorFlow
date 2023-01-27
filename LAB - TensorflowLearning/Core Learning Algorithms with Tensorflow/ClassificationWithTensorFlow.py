from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Defining the feature columns
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth','Species']
SPECIES = ['Setosa','Versicolor','Virginica']

#Using Keras (a subsidiary module from TensorFlow) to get the file we need's file path
train_path = tf.keras.utils.get_file(
  "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
  "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#Reading the datafile into a pandas dataframe
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES,header=0)

#Popping / Removing the Species Feature column from the main dataframes and 
# moving it into another dataframe that's meant to hold the desired outputs
train_y = train.pop('Species')
test_y = test.pop('Species')


def input_fn(features, labels, training=True, batch_size=256):
  #Convert the inputs to a dataset
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  #Shuffle and repeat if you are in training mode
  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
  my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#Build a DNN(Deep Neural Network)Classifier with 2 hidden layers with 30 and 10 hidden nodes/neurons each
classifier = tf.estimator.DNNClassifier(
  feature_columns=my_feature_columns,
  #Two hidden layers of 30 and 10 nodes respectively
  hidden_units=[30, 10],
  #The model must choose between 3 classes
  n_classes=3)

classifier.train(
  input_fn=lambda: input_fn(train,train_y,training=True),
  steps=5000)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test,test_y,training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def input_fn2(features, batch_size=256):
  #Convert the inputs to a dataset without labels
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth']
predict = {}

print("Please type numeric values as promted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False
  
  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn2(predict))
for pred_dict in predictions:
  class_id = pred_dict['class_ids'][0]
  probability = pred_dict['probabilities'][class_id]

  print('Prediction is "{}" ({:.1f}%)'.format(
    SPECIES[class_id], 100 * probability))