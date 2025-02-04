#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import tensorflow as tf
import numpy as np
import time

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))

#Operator overloading is also supported
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2,3]])
print(x)
print(x.shape)
print(x.dtype)

ndarray = np.ones([3,3])

print("TensorFlow operations convert numpy array to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

x = tf.random.uniform([3,3])

print("Is there a GPU Available: "),
print(tf.config.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0: ")
print(x.device.endswith('GPU:0'))


def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x,x)
    
  result = time.time()-start
  
  print("10 loops: {:0.2f}ms".format(1000*result))

#Force execution on cpu
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000,1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)
  
#Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): #Or GPU1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000,1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
    

ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

#Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
          Line 2
          Line 3
          """)
  
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print('Elements of ds_tensors: ')
for x in ds_tensors:
  print(x)
  
print('\nElements in ds_file')
for x in ds_file:
  print(x)