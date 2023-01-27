from sys import version
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

use_builtins = True

class ShapeChecker():

  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    if isinstance(names, str):
      names = (names,)

    shape = tf.shape(tensor)
    rank = tf.rank(tensor)

    if rank != len(names):
      raise ValueError(f'Rank mismatch:\n'
                       f'    found {rank}: {shape.numpy()}\n'
                       f'    expected {len(names)}: {names}\n')

    for i, name in enumerate(names):
      if isinstance(name, int):
        old_dim = name
      else:
        old_dim = self.shapes.get(name, None)
      new_dim = shape[i]

      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

"""
Download and prepare the dataset
"""
#Download the file
import pathlib

path_to_zip = tf.keras.utils.get_file(
  'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

def load_data(path):
  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  inp = [inp for targ, inp in pairs]
  targ = [targ for targ, inp in pairs]

  return targ, inp

targ, inp = load_data(path_to_file)
print(inp[-1])
print(targ[-1])

"""
Create a tf.data dataset
"""
BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

for example_input_batch, example_target_batch in dataset.take(1):
  print(example_input_batch[:5])
  print()
  print(example_target_batch[:5])
  break

"""
Text preprocessing
"""
#Standardization
example_text = tf.constant('¿Todavía está en casa?')

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())

def tf_lower_and_split_punct(text):
  #Split accecented characters.
  text = tf_text.normalize_utf8(text,'NFKD')
  text = tf.strings.lower(text)
  #Keep space, a to z and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  #Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  #Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())

"""
Text Vectorization
"""
max_vocab_size = 5000

input_text_processor = preprocessing.TextVectorization(
  standardize=tf_lower_and_split_punct,
  max_tokens=max_vocab_size)

input_text_processor.adapt(inp)

#Here are the first 10 words from the vocabulary:
print(input_text_processor.get_vocabulary()[:10])

output_text_processor = preprocessing.TextVectorization(
  standardize=tf_lower_and_split_punct,
  max_tokens=max_vocab_size)

output_text_processor.adapt(targ)
print(output_text_processor.get_vocabulary()[:10])

example_tokens = input_text_processor(example_input_batch)
print(example_tokens[:3, :10])


input_vocab = np.array(input_text_processor.get_vocabulary())
tokens = input_vocab[example_tokens[0].numpy()]
' '.join(tokens)

mpl.use('tkagg')

plt.subplot(1,2,1)
plt.pcolormesh(example_tokens)
plt.title('Tokens IDs')

plt.subplot(1,2,2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')
plt.show()

"""
The encoder/decoder model
"""
embedding_dim = 256
units = 1024
#The encoder
class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size

    #The embedding layer converts token to vectors
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

    #The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units, #Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    shape_checker = ShapeChecker()
    shape_checker(tokens, ('batch','s'))

    #2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    #3. The GRU processes the embedding sequence
    #   output shape: (batch, s, enc_units)
    #   state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    shape_checker(output, ('batch', 's', 'enc_units'))
    shape_checker(state, ('batch', 'enc_units'))

    #4. Returns the new sequence and its state.
    return output, state


#Here is how it fits together so far:
#Convert the input text to tokens.
example_tokens = input_text_processor(example_input_batch)

#Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)

example_enc_output, example_enc_state = encoder(example_tokens)

print(f'Input batch, shape (batch): {example_input_batch.shape}')
print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')
