import numpy as np
import tensorflow as tf

MAX=4800
# Selecting the data to be augmented and augmenting process
def get_augment_id(indexes,MAXIMUM = MAX):
    aug_id=[]#np.empty((0,1),dtype="int32")
    pixel_id = []
    for classes in indexes:
        if MAXIMUM - classes.shape[0] < classes.shape[0]:
            print("LESS : ",classes.shape[0])
            split_index = MAXIMUM-classes.shape[0]
            augmented_index = np.random.choice(classes,(split_index,),replace=False)
            aug_id.append(augmented_index[:split_index//2])#np.concatenate((aug_id,augmented_index),axis=0)
            pixel_id.append(augmented_index[split_index//2:])
        else:
            if MAXIMUM - (2*classes.shape[0]) < classes.shape[0]:
                print("MORE : ",classes.shape[0])
                split_index = MAXIMUM-(2*classes.shape[0])
                augmented_index = np.reshape(classes,(classes.shape[0],))
                aug_id.append(augmented_index[:split_index//2])#np.concatenate((aug_id,augmented_index),axis=0)
                pixel_id.append(augmented_index[split_index//2:])
            else:
                print("MORE : ",classes.shape[0])
                augmented_index = np.reshape(classes,(classes.shape[0],))
                aug_id.append(augmented_index)
                pixel_id.append(augmented_index)
    print()       
    return aug_id,pixel_id

#def get_sp_augment(num):
    
def get_pixel_augment(indexes,MAXIMUM = MAX):
    aug_id = []
    for classes in indexes:
        if MAXIMUM - (2*classes.shape[0]) < classes.shape[0]:
            augmented_index = np.random.choice(classes,(MAXIMUM-(2*classes.shape[0]),),replace=False)
            aug_id.append(augmented_index)#np.concatenate((aug_id,augmented_index),axis=0)
        else:
            augmented_index = np.reshape(classes,(classes.shape[0],))
            aug_id.append(augmented_index)#np.concatenate((aug_id,augmented_index),axis=0)
            
    return aug_id