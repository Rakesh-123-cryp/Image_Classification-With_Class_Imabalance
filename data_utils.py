import numpy as np

#Function to split test and train data
def data_split(indexes,percentage=0.1):
    """Not recommended but can be used if the validation set need not be distributed proportionally

    Args:
        indexes (list): list of indices of each class as a numpy.ndarray
        percentage (float, optional): the number of images to take for the validation set. Defaults to 0.1.

    Returns:
        numpy.ndarray: All the test indices
    """
    #looping through the classes to obtain the split
    test_list = np.empty((0,1))
    for cls in indexes:
        idx_count = int(np.floor(cls.shape[0]*percentage))
        
        idx_list = np.random.choice(cls,(idx_count,1),replace=False)
        test_list = np.concatenate((test_list, idx_list), axis=0)
    
    return test_list

def data_split_equal(indexes,per_class = 300):
    """This function returns the list of indexes for testing such 
    that equal images are taken from each class for validation

    Args:
        indexes (list): list of indices of each class as a numpy.ndarray
        per_class (int, optional): the number of images to take for the validation set. Defaults to 300.

    Returns:
        numpy.ndarray: All the test indices
    """
    test_list = np.empty((0,1))
    for cls in indexes:
        idx_list = np.random.choice(cls,(per_class,1),replace=False)
        test_list = np.concatenate((test_list, idx_list), axis=0)
        
    return test_list



#To limit the size of each class
def reduce(indexes,thresh = 4800):
    """Reduces the indices to the threshold given for each class (Must be lesser atmost the size of the max class)

    Args:
        indexes (list): list of indices of each class as numpy.ndarray

    Returns:
        list: returns the reduced form of the index list passed in
    """
    reduced_indexes = []
    for i in indexes:
        try:
            reduced_indexes.append(np.random.choice(i,size=(thresh,),replace=False))
        except:
            
            reduced_indexes.append(np.random.choice(i,size=(i.shape[0],),replace=False))
    
    return reduced_indexes