# Here we are creatining an image classifier for ladoo
#First of all we need to collect a lot of images for the ladoo dataset, for that we need to download different images of ladoos from  google 
#here i have kept 137 images for ladoos for training the data purpose
#make sure that all the data is there in a classified folder only meant for that purpose only Nou
#Now from the next line we will start doing it's code in python


#first we will import all the necessary python modules 
import numpy as np  #core library for scientific computing, contains powerful n-dimensional array object
import os           #the os module in python provides functions for interaction with the operating system.
from matplotlib import pyplot as plt   #visualization library in python for 2D plots of arrays
import cv2    #computer vision library
import random # importing "random" for random operations
import pickle # module translates an in-memory Python object into a serialized byte stream-a string of bytes that can be written to any file -like object

# it is my path where i have saved my images
Data_dir = "C:/Users/DELL/Downloads/Image_Recognition/data_2/" 

#here are all the categories you want your neural network to train
Categories = ["ganesh", "modak"]
  
#The size of the images must be fixed
img_size = 50

#now we will check for all the images in data_2(in my case the name of the folder in which i have saved my images is data_2) folder
for category in Categories:
  path = os.path.join(Data_dir, category) #here we have joined two folders using "os" folder
  for image in os.listdir(path): # here "os.listdir() method is used to get the list of all files and directories in the specified directory
    img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) # here "cv2.imread" method loads an image from the specified file and, "cv2.IMREAD_GRAYSCALE" is a flag it specifies to load the image in grayscale mode

#here will create a list 
training_data = []
def training_data():
  for category in Categories:
    path = os.path.join(Data_dir, category) #created path to "ganesh" and "ladoo"
    class_num = Categories.index(category) # get the classification (0 or 1)
    for image in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size)) #resize to normalize data size
        training_data.append([new_array, class_num])
      except Exception as e: #for the purpose of debugging and keeping the output clean
        pass
         
training_data() #calling the created function

# next we have to shuffl the data, right now our data is just all ganesh(lets say 0) and all ladoos(lets say 1) this will lead to error in the image classifier as the classifier will learn to just predict ganesh always then it
# will shift to ladoo going like this is not good either

random.shuffle(training_data)
#for testing that our data is shuffled properly we can write a code here
for sample in trainig_data[:5]:
  print(sample[1])
 
# now it's time to make our model
X = []  #features
Y = []  #label

for features, label in training_data:
  X.append(features)
  Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# now we have to save the data so that we can use it with the neural network model

pickle_out = open("X.pickel","wb") # The 'w' means that we are writing to the file, and b refers to binary mode
pickle.dump(X, pickle_out) # it takes two argument, the object we want to pickle and the file to which the object has to be saved
pickle_out.close() # it used to close the file which we have opened and it is important

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

# The process of loading the file back to the python program 
pickle_in = open("X.pickle", "rb") # The 'r' stands for read mode and 'b' stands for binary mode
X = pickle.load(pickle_in)

# The basic CNN stucture is as follows: Convolution--> Pooling-> Convolution --> Pooling--> Fully Connected Layer--> Output
# Convolution is the act of taking the original data, and creating feature maps from it.
# Pooling is down-sampling, most often in the form of "max_pooling", where we select a region, and then take the maximum value in that region
# Fully connected layers is the neural network where all the nodes are "fully connected".

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPlling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

#Opening the files about data
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

#normalizing the data as a pixle goes from 0 to 255
X = X/255

# now building the model
model = Sequential()
#here we are building a 3 convolution layer 
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


#2 Hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))
 
# The output layer with 2 neurons and 2 classes
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
# here we have used adam optimizer
model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, Y, batch_size = 32, epochs = 40, validation_split = 0.1)

#saving the model
mine_model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(mine_model_json)
    
model.save_weights("model.h5")
print("saved model to disk")

# it is a place where we have saved our model and we will be using it later
model.save('CNN.model')

# printing the graphs
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')

# predicting an image now 
# giving the path where our image is saved
# By following the same procedure that is being followed above that is first reading and resizing and reshaping the image to our desired dimension
import cv2
import tensorflow as tf

Categories = ["ganesh", "modak"]

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model") 
image = prepare("C:/Users/DELL/Pictures/Camera Roll/moda.jpg")
prediction = model.predict([image])
prediction = list(prediction[0])
print(Categories[prediction.index(max(prediction))])
                   

 
  
