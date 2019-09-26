#Importing Necessary Packages
import tensorflow as tf
from tensorflow import keras
import numpy as np

#Downloading the dataset
from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#Visualizing the Image Data
train_images.shape #(50000, 32, 32, 3)
test_images.shape #(10000, 32, 32, 3)

train_labels.shape #(50000, 1)
test_labels.shape #(10000, 1)

#Plotting the Image
img = train_images[19]

import matplotlib.pyplot as plt
plt.imshow(img )
plt.show

#Splitting the data into Train, Validation, Test Sets
val_images = train_images[:10000]
val_labels = train_labels[:10000]

train_images = train_images[10000:]
train_labels = train_labels[10000:]

print("Train_images Shape:", train_images.shape) #(40000, 32, 32, 3)
print("Train_labels Shape:", train_labels.shape) #(40000, 1)
print("Validation_images Shape:", val_images.shape) #(10000, 32, 32, 3)
print("Validation_labels Shape:", val_labels.shape) #(10000, 1)
print("Test_images Shape:", test_images.shape) #(10000, 32, 32, 3)
print("Test_labels Shape:", test_labels.shape) #(10000, 1)

#Data Preprocessing
from keras.utils import to_categorical

train_images = train_images.astype('float32')/255
val_images = val_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

#train_images = train_images.reshape((40000, 32, 32, 1))

print("Train_images Shape:", train_images.shape) #(40000, 32, 32, 3)
print("Train_labels Shape:", train_labels.shape) #(40000, 10)
print("Validation_images Shape:", val_images.shape) #(10000, 32, 32, 3)
print("Validation_labels Shape:", val_labels.shape) #(10000, 10)
print("Test_images Shape:", test_images.shape) #(10000, 32, 32, 3)
print("Test_labels Shape:", test_labels.shape) #(10000, 10)

#Defining the Conv Net
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

#Training and Validating the Model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_images, train_labels, epochs=8, batch_size=2048, validation_data=(val_images, val_labels))


#Continued.............