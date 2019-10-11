#Importing the Libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os, shutil
from zipfile import ZipFile
import cv2
import os
import glob


"""#Extracting the zip file"""

file_name = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall("/content/drive/My Drive/OASIS_dataset_tensorflow")
  print("Done")



"""#Importing the images and storing them in variables"""

#Directories of all images

img_seg_test_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_seg_test"
img_seg_train_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_seg_train"
img_seg_validate_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_seg_validate"
img_test_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_test"
img_train_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_train"
img_validate_dir = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data/keras_png_slices_validate"

#Segmented Test Images
data_path = os.path.join(img_seg_test_dir, '*g')
files = glob.glob(data_path)

seg_test = [] #Variables where all the seg_test images are saved

for f1 in files:
  print(len(seg_test))
  img = cv2.imread(f1)
  seg_test.append(img)
  
seg_test = np.array(seg_test) #Converting the list into a tensor

#Visualizing the shape and plotting the image (to cross check)

seg_test.shape #(544, 256, 256, 3)  cv2 grayscale
plt.imshow(seg_test[11,:,:,:])

#from skimage import color

#seg_test = color.rgb2gray(seg_test)
#seg_test.shape
#plt.imshow(seg_test[11,:,:])

np.save('seg_test.ipynb', seg_test)

#np.load('seg_test.ipynb.npy').shape

seg_test = np.load('seg_test.ipynb.npy')

seg_test.shape

#Segmented Train Images
data_path = os.path.join(img_seg_train_dir, '*g')
files = glob.glob(data_path)

seg_train = [] #Variables where all the seg_train images are saved

for f1 in files:
  print(len(seg_train))
  img = cv2.imread(f1)
  seg_train.append(img)
  
seg_train = np.array(seg_train) #Converting the list into a tensor

#Visualizing the shape and plotting the image (to cross check)

seg_train.shape #(9664, 256, 256, 3)  cv2 grayscale
plt.imshow(seg_test[1,:,:,:])

np.save('seg_train.ipynb', seg_train)

seg_train = np.load('seg_train.ipynb.npy')
seg_train.shape

#Segmented Validation Images
data_path = os.path.join(img_seg_validate_dir, '*g')
files = glob.glob(data_path)

seg_validate = [] #Variables where all the seg_validate images are saved

for f1 in files:
  print(len(seg_validate))
  img = cv2.imread(f1)
  seg_validate.append(img)
  
seg_validate = np.array(seg_validate) #Converting the list into a tensor

seg_validate.shape #(1120, 256, 256, 3)
plt.imshow(seg_validate[1,:,:,:])

np.save('seg_validate.ipynb', seg_validate)

seg_validate = np.load('seg_validate.ipynb.npy')
seg_validate.shape

#Test Images
data_path = os.path.join(img_test_dir, '*g')
files = glob.glob(data_path)

test = [] #Variables where all the test images are saved

for f1 in files:
  print(len(test))
  img = cv2.imread(f1)
  test.append(img)
  
test = np.array(test) #Converting the list into a tensor

test.shape #(544, 256, 256, 3)

np.save('test.ipynb', test)

test = np.load('test.ipynb.npy')
test.shape

#Train Images
data_path = os.path.join(img_train_dir, '*g')
files = glob.glob(data_path)

train = [] #Variables where all the train images are saved

i = 0

for f1 in files:
  #if i<3100:
    print(len(train))
    img = cv2.imread(f1)
    train.append(img)
    #i=i+1
  
  
train = np.array(train) #Converting the list into a tensor

train.shape #(9664, 256, 256, 3)

np.save('train.ipynb', train)

train = np.load('train.ipynb.npy')
train.shape

plt.imshow(train[199,:,:,:])

#Validate Images
data_path = os.path.join(img_validate_dir, '*g')
files = glob.glob(data_path)

validate = [] #Variables where all the test images are saved

for f1 in files:
  print(len(validate))
  img = cv2.imread(f1)
  validate.append(img)
  
validate = np.array(validate) #Converting the list into a tensor

np.save('validate.ipynb', validate)

validate = np.load('validate.ipynb.npy')
validate.shape

"""#Visualizing the data"""

#Shape of the tensors
print("Segmented Test Images shape:", seg_test.shape)
print("Segmented Train Images shape:", seg_train.shape)
print("Segmented Validate Images shape:", seg_validate.shape)
print("Test Images shape:", test.shape)
print("Train Images shape:", train.shape)
print("Validate Images shape:", validate.shape)

seg_test[1,100:200,100:200,1]

"""#Compiling the Model"""

#Data Preprocessing 
#Pixel values of Images are in the range 0-255 
#Pixel values of segmented Images are in the range 0-3

seg_test = seg_test.astype('float32')/85
seg_train = seg_train.astype('float32')/85
seg_validate = seg_validate.astype('float32')/85
test = test.astype('float32')/255
train = train.astype('float32')/255
validate = validate.astype('float32')/255

#Saving the data

np.save('seg_test.ipynb', seg_test)
np.save('seg_train.ipynb', seg_train)
np.save('seg_validate.ipynb', seg_validate)
np.save('test.ipynb', test)
np.save('train.ipynb', train)
np.save('validate.ipynb', validate)

#Loading the data  (session is crashing)

seg_test = np.load('seg_test.ipynb.npy')
seg_train = np.load('seg_train.ipynb.npy')
seg_validate = np.load('seg_validate.ipynb.npy')
test = np.load('test.ipynb.npy')
train = np.load('train.ipynb.npy')
validate = np.load('validate.ipynb.npy')

seg_test[1,100:200,100:200]

#Converting 3 channels to 1 channel

seg_test = seg_test[:,:,:,1]
seg_train = seg_train[:,:,:,1]
seg_validate = seg_validate[:,:,:,1]

test = test[:,:,:,1]
train = train[:,:,:,1]
validate = validate[:,:,:,1]

#Shape of the tensors
print("Segmented Test Images shape:", seg_test.shape)
print("Segmented Train Images shape:", seg_train.shape)
print("Segmented Validate Images shape:", seg_validate.shape)
print("Test Images shape:", test.shape)
print("Train Images shape:", train.shape)
print("Validate Images shape:", validate.shape)

seg_test[1,100:200,100:200]
plt.imshow(seg_test[1,:,:])

#Saving the data

np.save('seg_test.ipynb', seg_test)
np.save('seg_train.ipynb', seg_train)
np.save('seg_validate.ipynb', seg_validate)
np.save('test.ipynb', test)
np.save('train.ipynb', train)
np.save('validate.ipynb', validate)

from keras.utils import to_categorical

seg_test = to_categorical(seg_test)
#seg_train = to_categorical(seg_train)
seg_validate = to_categorical(seg_validate)

#Saving the Data

np.save('seg_test.ipynb', seg_test)
#np.save('seg_train.ipynb', seg_train)
np.save('seg_validate.ipynb', seg_validate)

#Loading the data  (session is crashing)

seg_test = np.load('seg_test.ipynb.npy')
seg_train = np.load('seg_train.ipynb.npy')
seg_validate = np.load('seg_validate.ipynb.npy')
test = np.load('test.ipynb.npy')
train = np.load('train.ipynb.npy')
validate = np.load('validate.ipynb.npy')

seg_test.shape
plt.imshow(seg_test[1,:,:,3])

from keras.utils import to_categorical

seg_train = to_categorical(seg_train)

np.save('seg_train.ipynb', seg_train)

#Expanding the dimension

test = np.expand_dims(test, axis=3)
train = np.expand_dims(train, axis=3)
validate = np.expand_dims(validate, axis=3)

np.save('test.ipynb', test)
np.save('train.ipynb', train)
np.save('validate.ipynb', validate)

#Loading the data  (session is crashing)

seg_test = np.load('seg_test.ipynb.npy')
seg_train = np.load('seg_train.ipynb.npy')
seg_validate = np.load('seg_validate.ipynb.npy')
test = np.load('test.ipynb.npy')
train = np.load('train.ipynb.npy')
validate = np.load('validate.ipynb.npy')

#Shape of the tensors
print("Segmented Test Images shape:", seg_test.shape)
print("Segmented Train Images shape:", seg_train.shape)
print("Segmented Validate Images shape:", seg_validate.shape)
print("Test Images shape:", test.shape)
print("Train Images shape:", train.shape)
print("Validate Images shape:", validate.shape)

plt.imshow(test[9,:,:,0])

"""#Building the UNET Model (Functional API)"""

import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras




initializer = tf.random_normal_initializer(0., 0.02)
input_size = (256,256,1)

#Encoder
#Downsampling

inputs = Input(input_size) # (bs, 256, 256, 1)
down1 = Conv2D(16, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(inputs) # (bs, 128, 128, 16)
down2 = Conv2D(32, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down1) # (bs, 256, 256, 3)
down3 = Conv2D(64, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down2) # (bs, 256, 256, 3)
down4 = Conv2D(128, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down3) # (bs, 256, 256, 3)
down5 = Conv2D(256, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down4) # (bs, 256, 256, 3)
down6 = Conv2D(512, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down5) # (bs, 256, 256, 3)
down7 = Conv2D(512, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down6) # (bs, 256, 256, 3)

down8 = Conv2D(1024, 4, strides=2, padding = 'same', activation = 'relu', use_bias=False, kernel_initializer = initializer)(down7) # (bs, 256, 256, 3)
#Decoder
#Upsampling and adding skip connections

up7 = Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(down8) # (bs, 256, 256, 3)
merge7 = concatenate([down7, up7])

up6 = Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge7) # (bs, 256, 256, 3)
merge6 = concatenate([down6, up6])

up5 = Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge6) # (bs, 256, 256, 3)
merge5 = concatenate([down5, up5])

up4 = Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge5) # (bs, 256, 256, 3)
merge4 = concatenate([down4, up4])

up3 = Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge4) # (bs, 256, 256, 3)
merge3 = concatenate([down3, up3])

up2 = Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge3) # (bs, 256, 256, 3)
merge2 = concatenate([down2, up2])

up1 = Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge2) # (bs, 128, 128, 16)
merge1 = concatenate([down1, up1]) # (bs, 128, 128, 32)

conv1 = Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu', use_bias=False, kernel_initializer=initializer)(merge1) # (bs, 256, 256, 16)
conv2 = Conv2D(8, 4, padding='same', activation='relu', kernel_initializer=initializer )(conv1)
conv3 = Conv2D(4, 1, activation='softmax')(conv2)
model = Model(input = inputs, output = conv3)

model.summary()

#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['mean_squared_error'])

#model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics=['mean_squared_error'])
model.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#history = model.fit(train, seg_train, epochs=1, batch_size=128, validation_data=(validate, seg_validate))
history = model.fit(train, seg_train, epochs=11, batch_size=128, validation_data=(validate, seg_validate))

test_loss, test_acc = model.evaluate(test, seg_test)

test_acc

"""#Saving the model"""

model.save('UNET.h5')

"""#Displaying the curves of loss and accuracy"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""#Making Predictions on the Test Set"""

x = model.predict(test)
x.shape

plt.imshow(x[9,:,:,1])
plt.show()

plt.imshow(seg_test[9,:,:,1])
plt.show()