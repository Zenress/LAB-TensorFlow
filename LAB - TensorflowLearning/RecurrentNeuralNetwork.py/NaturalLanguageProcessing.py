#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

from keras.datasets import imdb
from keras_preprocessing import sequence
from keras import preprocessing
import tensorflow as tf
import numpy as np
import keras

#Assigning the size of the vocabulary, meaning the amount of different words
VOCAB_SIZE = 88584

#Assigning the maximum length of characters for one review
MAXLEN = 250
#How large of a batch we will split into during training
BATCH_SIZE = 64

#Loading only 88584 words of the IMDB dataset
(train_data, train_labels),(test_data,test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

#Padding the data so that all of the reviews are 250 characters long. This is done by adding 0's to the start of the reviews
train_data = sequence.pad_sequences(train_data, MAXLEN) #Preprocessing the data so that each review is 250 characters long
test_data = sequence.pad_sequences(test_data, MAXLEN) #Preprocessing the data so that each review is 250 characters long

#Loading the model that i saved so i wouldn't have to train it constantlty
model = tf.keras.models.load_model("languageProcessing.h5")

results = model.evaluate(test_data, test_labels)

#Encoding Function
word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)

#Decoding Function
reverse_word_index = {value: key for (key,value) in word_index.items()}

def decode_integers(integers):
  PAD = 0
  text = ""
  for num in integers:
    if num != PAD:
      text += reverse_word_index[num] + " "

  return text[:-1]

print(decode_integers(encoded))

#Prediction Function
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred)
  print(result[0])

positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review)

negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)