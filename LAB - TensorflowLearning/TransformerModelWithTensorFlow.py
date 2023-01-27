#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#Standard imports ^

import collections
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras.layers.normalization

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

"""
Download the Dataset
"""
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

for pt_examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    print(en.decode('utf-8'))


"""
Text tokenization & detokenization
"""
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
  f"{model_name}.zip",
  f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
  cache_dir='.', cache_subdir='.', extract=True
)

#The tf.saved_model contains two text tokenizers, one for English and one for Portuguese. Both have the same methods:
tokenizers = tf.saved_model.load(model_name)
[item for item in dir(tokenizers.en) if not item.startswith('_')]

"""
The tokenize method converts a bat ch of strings to a padded-batch of token IDs. This method splits punctuation, lowercases and uncode-normalizes the input before tokenizing. That standardization is not visible here because the input data is already standardized
"""
encoded = tokenizers.en.tokenize(en_examples)

#The detokenize method attempts to convert these token IDs back to human readabletext
round_trip = tokenizers.en.detokenize(encoded)

#The lower level lookup method converts from token-IDs to token text:
tokens = tokenizers.en.lookup(encoded)

"""
Setup Input pipeline
"""
def tokenize_pairs(pt,en):
  pt = tokenizers.pt.tokenize(pt)
  #Convert from ragged to dense, padding with zeros
  pt = pt.to_tensor()

  en = tokenizers.en.tokenize(en)
  #Convert from ragged to dense, padding with zeros
  en = en.to_tensor()
  return pt,en

#Here's a simple input pipeline that processes, shuffles and batches the data:
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
    ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

"""
Positional encoding
"""
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000,(2 *(i//2)) / np.float32(d_model))
  return pos * angle_rates  

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  #Apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  #Apply cvos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

#Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2,1,0))
pos_encoding = tf.reshape(pos_encoding, (d,n))

plt.pcolormesh(pos_encoding, cmap="RdBu")
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

"""
Masking
"""
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  #Add extra dimensions to add the padding
  #To the attention logits
  return seq[:, tf.newaxis, tf.newaxis, :] #(batch_size, 1, 1, seq_len)

x = tf.constant([[7,6,0,0,1], [1,2,3,0,0],[0,0,0,4,5]])
create_padding_mask(x)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1,0)
  return mask #(seq_len, seq_len)

x = tf.random.uniform((1,3))
temp = create_look_ahead_mask(x.shape[1])

"""
Scaled dot product attention
"""
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimensions, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.
  
  Args:
   q: query shape == (..., seq_len_q, depth)
   k: key shape == (..., seq_len_k, depth)
   v: value shape == (..., seq_len_vm depth_v)
   mask: Float tensor with shape broadcastable
         to (..., seq_len_q, seq_len_k). Defaults to None.
           
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len_q, seq_len_k)

  #Scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  #Add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask + -1e9)

  #Softmax is normalized on the last axis (seq_len_k) so that the scores
  #Add up to 1
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=1) #(..., seq_len_k, seq_len_k)
  output = tf.matmul(attention_weights, v) #(..., seq_len_q, depth_v)

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
    q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32) #(4,3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)
#This 'query' aligns with the second 'key',
# So the second 'value' is returned.
temp_q = tf.constant([[0,10,0]], dtype=tf.float32) #(1,3)
print_out(temp_q, temp_k, temp_v)

# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)

"""
Multi-head attention
"""
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.reshape(q)[0]

    q = self.wq(q) # (batch_size, seq_len, d_mmodel)
    k = self.wq(k) # (batch_size, seq_len, d_mmodel)
    v = self.wq(v) # (batch_size, seq_len, d_mmodel)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1,60,512)) #(batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
