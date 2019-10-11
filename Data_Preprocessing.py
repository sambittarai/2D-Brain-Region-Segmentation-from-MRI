#Loading the data  (session is crashing)

seg_test = np.load('seg_test.ipynb.npy')
seg_train = np.load('seg_train.ipynb.npy')
seg_validate = np.load('seg_validate.ipynb.npy')
test = np.load('test.ipynb.npy')
train = np.load('train.ipynb.npy')
validate = np.load('validate.ipynb.npy')

#Data Preprocessing 
#Pixel values of Images are in the range 0-255 
#Pixel values of segmented Images are (0, 85, 170, 255)

#Normalizing the Data 
#After normalizing MRI images are in the range 0-1 and their corresponding segmented images are in the range 0-3
seg_test = seg_test.astype('float32')/85
seg_train = seg_train.astype('float32')/85
seg_validate = seg_validate.astype('float32')/85
test = test.astype('float32')/255
train = train.astype('float32')/255
validate = validate.astype('float32')/255


#Converting 3 channels to 1 channel

seg_test = seg_test[:,:,:,1]
seg_train = seg_train[:,:,:,1]
seg_validate = seg_validate[:,:,:,1]

test = test[:,:,:,1]
train = train[:,:,:,1]
validate = validate[:,:,:,1]

#Creating the masks for the segmented images
from keras.utils import to_categorical

seg_test = to_categorical(seg_test)
seg_train = to_categorical(seg_train)
seg_validate = to_categorical(seg_validate)

#Expanding the dimension

test = np.expand_dims(test, axis=3)
train = np.expand_dims(train, axis=3)
validate = np.expand_dims(validate, axis=3)

#Shape of the tensors
print("Segmented Test Images shape:", seg_test.shape)
print("Segmented Train Images shape:", seg_train.shape)
print("Segmented Validate Images shape:", seg_validate.shape)
print("Test Images shape:", test.shape)
print("Train Images shape:", train.shape)
print("Validate Images shape:", validate.shape)