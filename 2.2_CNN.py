#Import Necessary Libraries
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Load the data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

lfw_people.images.shape #Shape of the image and shape of the image

X = lfw_people.data 
y = lfw_people.target 

print(X.shape) #(1288,1850)
print(y.shape) #(1288,)

#Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Data Preprocessing
print("Image Size:", lfw_people.images.shape)
print("Current Image size:", X_train.shape)
X_test.shape

from keras.utils import to_categorical

X_train = X_train.reshape((966, 50, 37))
X_train = X_train.astype('float32')/255

X_test = X_test.reshape((322, 50, 37))
X_test = X_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Image Visualization
X_train[1].shape #(50, 37)

img = X_train[9]

import matplotlib.pyplot as plt
plt.imshow(img )
plt.show

#Reshaping the Images
X_train = X_train.reshape((966, 50, 37, 1))
X_test = X_test.reshape((322, 50, 37, 1))

#Defining the Conv Net
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(50,37,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))

#Adding a classifier

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()

#Training
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

from keras import optimizers

history = model.fit(X_train, y_train, epochs=20, batch_size=16)

#Testing
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test_acc:", test_acc)

#Displaying curves of loss and accuracy
from keras.preprocessing.image import ImageDataGenerator
acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, loss, label='Training Loss')
plt.show()