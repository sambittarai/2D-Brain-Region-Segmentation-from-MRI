#2.1 (FFNN)

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np 
import tensorflow as tf
from tensorflow import keras

#Load the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

lfw_people.images.shape #Shape of the image and shape of the image

X = lfw_people.data 
y = lfw_people.target 
#y -> we have 7 labels from 0-6

#Splitting the data into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Visualizing the data (size)
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

#Normalizing the Image Data
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

#Converting the Labels into one hot vector
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[9,:] #Testing 

#Network Architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(1850,)))
network.add(layers.Dense(64, activation='relu'))
#network.add(layers.Dense(32, activation='relu'))

network.add(layers.Dense(7, activation='softmax'))

#Network Compilation
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting the Network
network.fit(X_train, y_train, epochs=20, batch_size=8)

#Network Performance on the testing set
test_loss, test_acc = network.evaluate(X_test, y_test)
print('Test_acc:', test_acc)