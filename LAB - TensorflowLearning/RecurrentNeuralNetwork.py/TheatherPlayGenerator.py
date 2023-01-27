#Switching CPU operation instructions to AVX AVX2
import os

from tensorflow.python.keras.layers.recurrent import RNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import numpy as np

#REGION: Readying the data
#Getting the dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#Read, then decode for py2 compat
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
#Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
#REGION END: Readying the data

#REGION: Processing the data
#from text to int
def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

#from int to text again
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

#Splitting the data up so that it's easier to handle
seq_length = 100 #Length of a sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)
#REGION END: Processing the data

#REGION: Training DATA
#Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1,drop_remainder=True)


def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text,target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) #Vocab is the number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

#Buffer size to shuffle the dataset
#(TF Data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#REGION END: Training DATA

#REGION: Model creation
#Creating a model that can later be rebuild to be used to predict on 1 piece of data instead of a ton of data
def build_model(vocab_size,embedding_dim,rnn_units,batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,
                              batch_input_shape=[batch_size,None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
#REGION END: Model creation

#REGION: Building the model
model = build_model(VOCAB_SIZE,EMBEDDING_DIM,RNN_UNITS,BATCH_SIZE)
#REGION END: Building the model

#REGION: Creating a loss function
#The reason for creating a loss function ourselves is that there is no built-in loss function that can handle 3 dimensional nested arrays
def loss(labels,logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
#REGION END: Creating a loss function

#REGION: Compiling the model
model.compile(optimizer='adam',loss=loss)
#REGION END: Compiling the model

#REGION: Setting up checkpoints?
#Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
#Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath = checkpoint_prefix,
  save_weights_only=True)
#REGION END:

#REGION: Training the Model
history = model.fit(data,epochs=2,callbacks=[checkpoint_callback])
#REGION END: Training the Model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM,RNN_UNITS,batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) #Loading the latest checkpoint
model.build(tf.TensorShape([1,None])) #Expect 1 input. We don't know the next layers dimensions


def generate_text(model,start_string):
  #Evaluation step (generating text using the learned model)
  #Number of characters to generate
  num_generate = 800

  #Converting our start string to number (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval,0)

  #Empty string to store our results
  text_generated = []

  #Low temperatures results in more predictable text.
  #Higher temperatures results in more surprising text
  #Experiment to find the best setting
  temperature = 1.0

  #Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    #Remove the batch dimension
    predictions = tf.squeeze(predictions,0)

    #Using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

    #We pass the predicted character as the next input to the model
    #Along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id],0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model,inp))