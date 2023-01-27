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

#Defining feature columns
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Making the model a sequential model
model = models.Sequential() 
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3))) #Defining that the first layer after input is a conv2D layer
model.add(layers.MaxPooling2D((2,2))) #Defining that the second layer after the input is a Maxpooling2D layer
model.add(layers.Conv2D(64,(3,3), activation='relu')) #Defining that the third layer after the input is a Conv2D layer
model.add(layers.MaxPooling2D((2,2))) #Defining that the fourth layer after the input is a Maxpooling2D layer
model.add(layers.Conv2D(64,(3,3), activation='relu')) #Defining that the fifth layer after the input is a Conv2D layer
#Adding Dense Layers / These layers are added to add a way to classify the previous output from the Conv2D layer above
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10))
#The above model is split into: Convolutional base and Classifier. The comment with: Adding Dense layers is also the split line between these two


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images,train_labels, epochs=10,
                    validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)
print(test_acc)
