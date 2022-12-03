from __future__ import annotations
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\cudnn-windows-x86_64-8.6.0.163_cuda11-archive\bin")
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Flatten, Conv2D, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from sre_parse import CATEGORIES
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tqdm import tqdm

DATADIR  = 'C:\Users\spike\Pictures\Train'
Categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K","L",'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'Nothing', 'del', 'Space']
minValue = 70
img_size=128
for category in Categories:
    path = os.path.join(DATADIR,category)
    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path,image)) 
        gray=cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
        cv2.imshow("thresh", res)
        
        break

        
    break
print(image_array)
print(image_array.shape)

IMG_SIZE = 128


new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
training_data = []
def createtraining_data():
    for category in Categories:
        path = os.path.join(DATADIR,category)
        classes = Categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                image_array = cv2.imread(os.path.join(path,image))
                gray=cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(5,5),2)
                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
           
                new_array = cv2.resize(res, (IMG_SIZE, IMG_SIZE))  
                training_data.append([new_array, classes])
            except Exception as e:
                pass

createtraining_data()
print(len(training_data))

import random

random.shuffle(training_data)

for sample in training_data[:20]:
    print(sample[1])

X = []
y = []



for features,label in training_data:
    X.append(features)
    y.append(label)




X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X_data = X
y_data = y


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 


from keras.utils import to_categorical
y_cat_train = to_categorical(y_train,30)
y_cat_test = to_categorical(y_test,30)

def alexnet(input_shape = (128, 128, 1), n_classes = 30):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape = input_shape))
    
   
    model.add(Conv2D(48, (5, 5), activation = 'relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    
    
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(192, (2, 2), activation = 'relu'))
    
   
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
  
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(192, (2, 2), activation = 'relu'))
    

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (2, 2), activation = 'relu'))
    

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    
    
    model.add(Dense(units = 1024, activation = 'relu'))
    model.add(Dropout(rate=0.40))
    
    model.add(Dense(units = 1024, activation = 'relu'))
    
    model.add(Dropout(rate=0.40))
    
    model.add(Dense(n_classes, activation = 'softmax'))
    return model

AlexNet = alexnet()




AlexNet.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              


AlexNet.fit(X_train, y_cat_train,
          epochs=20,
          batch_size=64,
          verbose=2,
          validation_data=(X_test, y_cat_test))

AlexNet.save('hand_model.h5')