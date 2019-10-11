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
