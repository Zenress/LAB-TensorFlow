#Switching CPU operation instructions to AVX AVX2
import os
from sys import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Main Machine learning libraries
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #Split into testing and training
 
class_names = ['T-shirt/top', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential ([
    keras.layers.Flatten(input_shape=(28,28)), #Input layer (1) / 28x28 neurons = 784
    keras.layers.Dense(128, activation='relu'), #Hidden layer (2) Method used: REctifiedLinearUnit / 128 neurons
    keras.layers.Dense(10, activation='softmax') #Output layer (3) Method used: Softmax / 10 neurons
])

model.compile(optimizer='adam', #Method used to change weights and bias
    loss='sparse_categorical_crossentropy', #This is the cost function that is used to calculate how much the given weight and bias needs to change
    metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=2) #Passing the training data through, and running through it 10 times (epochs)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=1) # Testing for the real accuracy using the validation set / testing set

print('Test accuracy:', test_acc)  

predictions = model.predict(test_images)
print(predictions)

#Script by TIM
#
COLOR = 'blue'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
#
def predict(model, image, correct_label): #Method for predicting the result
   class_names = ['T-shirt/top', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
   prediction = model.predict(np.array([image]))
   predicted_class = class_names[np.argmax(prediction)]
   show_image(image, class_names[correct_label], predicted_class)
#
def show_image(img,label,guess): #Method for plotting the results
   plt.figure()
   plt.imshow(img,cmap=plt.cm.binary)
   plt.title("Expected: "+ label)
   plt.xlabel("Guess: "+ guess)
   plt.colorbar()
   plt.grid(False)
   plt.show()
#
def get_number(): #Method for figuring out which index to use as the predicted number
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
      else:
        print("Try again...")
#
num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
# End of the script made by TIM