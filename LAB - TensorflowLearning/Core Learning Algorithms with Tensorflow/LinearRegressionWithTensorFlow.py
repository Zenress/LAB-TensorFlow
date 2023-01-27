import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column
import urllib
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Loading Dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Training Dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Eval / Testing Dataset
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#Figuring out all of the non nummeric columns AKA Categorical columns and assigning their names to a variable
CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
#Figuring out all of the numeric columns and assigning their name to a variable
NUMERIC_COLUMNS = ['age','fare']

feature_columns = [] #Creating a feature column variable that stores feature columns
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() #Gets a list of all the unique values from the currently iterated feature column.
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary)) #Adds a value to the array with the name of the currently iterated column + it's unique values

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float64)) #Adds a value to the Array with the name of the column and what datatype it is

print(feature_columns)

def make_input_fn(data_df,label_df,num_epochs=10, shuffle=True,batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval,y_eval, num_epochs=1, shuffle=False)

#Creating the Linear Regression Model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#Training the model
linear_est.train(train_input_fn)
#Evaluating the result
result = linear_est.evaluate(eval_input_fn)

#Printing the results accuracy
print(result['accuracy'])

