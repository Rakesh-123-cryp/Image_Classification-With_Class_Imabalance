import tensorflow as tf
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import random
import os

class dataLoader(tf.keras.utils.Sequence):
    def __init__(self,path_list, train_path,images_class,indices, spatial_index, pixel_index, batch_size=64,shuffle=True):
        self.shuffle=True
        self.index = indices
        self.path_list = path_list
        self.classes = images_class
        self.batch_size = batch_size
        self.ratio = spatial_index.shape[0]+indices.shape[0]+pixel_index.shape[0]
        #self.size = (int(np.floor(ratio*batch_size+0.5)),int(np.floor((1-ratio)*batch_size+0.5)))
        self.path = train_path
        self.spatial_index = spatial_index
        self.pixel_index = pixel_index
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(self.index.shape[0]/self.batch_size))

    def __getitem__(self, step):
        main_size = int(np.ceil(self.index.shape[0]/self.ratio+0.5))
        index = self.index[step*(main_size):(step+1)*(main_size)]
        
        spatial_size = int(np.ceil(self.spatial_index.shape[0]/self.ratio+0.5))
        aug_index = self.spatial_index[step*(spatial_size):(step+1)*(spatial_size)]
        
        pixel_size = int(np.ceil(self.pixel_index.shape[0]/self.ratio+0.5))
        pixel_index = self.pixel_index[step*(spatial_size):(step+1)*(spatial_size)]
        
        images = []

        #Each image path is read and resized
        for img in self.path_list[0,index]:
            read_image = cv.resize(cv.imread(os.path.join(self.path,img)),(224,224),interpolation = cv.INTER_AREA)
            read_image = np.divide(read_image,255)
            images.append(read_image)
        
        #shaping the array
        images = np.array(images)

        #Spatial Augmenting the data
        spatial,pixel = self.get_augment_pipeline()
        for img in self.path_list[0,aug_index]:
            read_image = cv.resize(cv.imread(os.path.join(self.path,img)),(224,224),interpolation = cv.INTER_AREA)
            read_image = spatial.flow(np.reshape(read_image,(1,)+read_image.shape))
            read_image = read_image.__getitem__(0)
            read_image = np.divide(read_image,255)
            images = np.concatenate((images,read_image),axis=0)
        
        #Pixel Augmenting the data
        for img in self.path_list[0,pixel_index]:
            read_image = cv.resize(cv.imread(os.path.join(self.path,img)),(224,224),interpolation = cv.INTER_AREA)
            read_image = pixel.flow(np.reshape(read_image,(1,)+read_image.shape))
            read_image = read_image.__getitem__(0)
            read_image = np.divide(read_image,255)
            images = np.concatenate((images,read_image),axis=0)
            
        #Getting the Labels
        labels = np.concatenate((self.classes[index],self.classes[aug_index],self.classes[pixel_index]),axis=0)
        processed_label = tf.one_hot(labels,7)
        
        return images,tf.reshape(processed_label,(processed_label.shape[0],7))
        
    def on_epoch_end(self):
        np.random.shuffle(self.index)
        np.random.shuffle(self.spatial_index)
        np.random.shuffle(self.pixel_index)
    
    def get_augment_pipeline(self):
        #Augmentation Pipeline
        spatial_augment_pipeline = ImageDataGenerator(rotation_range=90,width_shift_range=0.1,height_shift_range=0.1)#,preprocessing_function=self._get_preprocessing_function)
        pixel_augment_pipeline = ImageDataGenerator(brightness_range=[0.1,0.2],zca_whitening=True,channel_shift_range=0.1)
        return spatial_augment_pipeline, pixel_augment_pipeline
    
    def _get_preprocessing_function(self,image):
        variability = np.random.randint(10,20)
        std = variability*random.random()
        noise = np.random.normal(0,std,image.shape)
        image = image+noise
        np.clip(image,0.0,255.0)
        image /= 255.0
        return image
    
class testLoader(tf.keras.utils.Sequence):
    def __init__(self, path_list, test_path, images_class, index, batch_size=32, shuffle=True):
        self.path_list = path_list
        self.index = index
        self.batch_size = batch_size
        self.size = int(np.ceil(index.shape[0]/batch_size+0.5))
        self.classes = images_class
        self.path = test_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.index.shape[0]/self.batch_size))

    def __getitem__(self, step):
        index = self.index[step*self.size:(step+1)*self.size,0]
        paths_to_read = self.path_list[0,index]
        images=[]
        #Each image path is read and resized
        for img in paths_to_read:
            read_image = cv.resize(cv.imread(os.path.join(self.path,img)),(224,224),interpolation = cv.INTER_AREA)
            read_image = np.divide(read_image,255)
            images.append(read_image)

        #shaping the array (Preprocessing)
        images = np.array(images)
        

        #Getting the Labels
        labels = np.concatenate((self.classes[index]),axis=0)
        processed_label = tf.one_hot(labels,7)

        return images, tf.reshape(processed_label,(processed_label.shape[0],7))

    def on_epoch_end(self):
        np.random.shuffle(self.index)