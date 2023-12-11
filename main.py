import os               # to collect the datafrom folder
import cv2              #to read images 
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical

import tensorflow as ts
from tensorflow import _keras
image_directory='dataset/'              # To insert path
no_hemorrhage_images=os.listdir(image_directory+'no/')       #to create the list for all "no" images
yes_hemorrhage_images=os.listdir(image_directory+'yes/')     #to create the list for all "yes" images
dataset=[]
label=[]
IMPUT_SIZE=64
# print(no_hemorrhage_images)

for i, image_name in enumerate(no_hemorrhage_images):        
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((IMPUT_SIZE,IMPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_hemorrhage_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((IMPUT_SIZE,IMPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
print(len(dataset))
print(len(label))
# Now to convert the dataset into numpy array
dataset=np.array(dataset)
label=np.array(label)
x_train, x_test, y_train, y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)
x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

# Model Building
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(IMPUT_SIZE,IMPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# to make all images into one factor i.e. to make them linear
model.add(Flatten())
model.add(Dense(64))     
model.add(Activation('relu'))   # Activation functiion   #sigmaodal
model.add(Dropout(0.5))         # To minimize the bar fitting problem
model.add(Dense(2))             # To add output layer
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10, validation_data=(x_test,y_test),shuffle=False)
model.save('model.keras')
