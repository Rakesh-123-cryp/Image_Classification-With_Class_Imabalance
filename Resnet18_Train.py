import numpy as np
import pandas as pd
import glob as glob
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras import layers
import keras
import tensorflow as tf
from glob import glob
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, InputLayer, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from ResNet18 import ResidualNetwork18
import random
from data_utils import reduce, data_split_equal
from Augmentation_util import get_augment_id
from data_Loaders import dataLoader,testLoader
import copy

path="/Users/rakesh/Downloads/fashion_images/train"
path_list = np.array(os.listdir(path),ndmin=2)
test_path = "/Users/rakesh/Downloads/fashion_images/test"

df = pd.read_csv("/Users/rakesh/Downloads/fashion_images/train.csv")
images_class = df["label"].values

# Reshaping the classes for training
images_class = np.reshape(images_class,(images_class.shape[0],1))
print("Extracted X paths and Y\n","*"*50,"\n")



#Getting the indexes of different classes
indexes=[]
for i in range(7):
    indexes.append(np.argwhere(images_class==i)[...,0])
indexes = reduce(indexes)

#Debug statement
for i in indexes:
    print(i.shape)

MAXIMUM = 4800
SPLIT_SIZE = 300
#Calling test split for images
test_index = data_split_equal(indexes,SPLIT_SIZE).astype("int64")

#Train split of images
indexes_copy = copy.deepcopy(np.concatenate(indexes,axis=0))
train_index = np.setdiff1d(indexes_copy,test_index)
print(np.unique(images_class[train_index],return_counts=True))

for i in range(len(indexes)):
    indexes[i] = np.setdiff1d(indexes[i],test_index[i*SPLIT_SIZE:(i+1)*SPLIT_SIZE])
    print(indexes[i].shape)
    
#The two different augmenting methods
spatial_aug_id, pixel_aug_id = get_augment_id(indexes,MAXIMUM=MAXIMUM)
spatial_aug_id = np.concatenate(spatial_aug_id,axis=0)
pixel_aug_id = np.concatenate(pixel_aug_id,axis=0)

indexes = np.concatenate(indexes,axis=0)

if __name__ == "__main__":
    
    trainLoader = dataLoader(path_list,path,images_class,train_index,spatial_aug_id,pixel_aug_id,batch_size=128)
    valLoader = testLoader(path_list,path,images_class,test_index,batch_size=128)

    print("Final Shape : ",np.unique(images_class[np.concatenate((indexes,spatial_aug_id,pixel_aug_id),axis=0),0],return_counts=True))
    #Class Weights
    ratios = class_weight.compute_class_weight("balanced",classes = np.unique(images_class),y=images_class[np.concatenate((indexes,spatial_aug_id,pixel_aug_id),axis=0),0])
    ratio_dict = dict(enumerate(ratios))

    # Optimizer
    opt = tf.keras.optimizers.legacy.Adam()

    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
    model = ResidualNetwork18(7)
    model.build(input_shape = (None,224,224,3))
    model.compile(optimizer = opt,loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    model.fit(trainLoader,epochs=20,batch_size=128,class_weight=ratio_dict,shuffle=True,callbacks=[callback],validation_data=valLoader)

    model.save("model.keras")