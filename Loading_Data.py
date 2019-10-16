from zipfile import ZipFile

"""#Extracting the zip file"""

file_name = "/content/drive/My Drive/OASIS_dataset_tensorflow/keras_png_slices_data.zip"



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
  img = cv2.imread(f1)
  seg_test.append(img)
  
seg_test = np.array(seg_test) #Converting the list into a tensor


#Segmented Train Images
data_path = os.path.join(img_seg_train_dir, '*g')
files = glob.glob(data_path)

seg_train = [] #Variables where all the seg_train images are saved

for f1 in files:
  img = cv2.imread(f1)
  seg_train.append(img)
  
seg_train = np.array(seg_train) #Converting the list into a tensor


#Segmented Validation Images
data_path = os.path.join(img_seg_validate_dir, '*g')
files = glob.glob(data_path)

seg_validate = [] #Variables where all the seg_validate images are saved

for f1 in files:
  img = cv2.imread(f1)
  seg_validate.append(img)
  
seg_validate = np.array(seg_validate) #Converting the list into a tensor


#Test Images
data_path = os.path.join(img_test_dir, '*g')
files = glob.glob(data_path)

test = [] #Variables where all the test images are saved

for f1 in files:
  img = cv2.imread(f1)
  test.append(img)
  
test = np.array(test) #Converting the list into a tensor

test.shape #(544, 256, 256, 3)


#Train Images
data_path = os.path.join(img_train_dir, '*g')
files = glob.glob(data_path)

train = [] #Variables where all the train images are saved

i = 0

for f1 in files:
    img = cv2.imread(f1)
    train.append(img)
  
  
train = np.array(train) #Converting the list into a tensor

train.shape #(9664, 256, 256, 3)



#Validate Images
data_path = os.path.join(img_validate_dir, '*g')
files = glob.glob(data_path)

validate = [] #Variables where all the test images are saved

for f1 in files:
  img = cv2.imread(f1)
  validate.append(img)
  
validate = np.array(validate) #Converting the list into a tensor


#Saving the data
np.save('seg_test.ipynb', seg_test)
np.save('seg_train.ipynb', seg_train)
np.save('seg_validate.ipynb', seg_validate)
np.save('test.ipynb', test)
np.save('train.ipynb', train)
np.save('validate.ipynb', validate)