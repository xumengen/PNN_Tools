import numpy as np

def compute_l2_norm(array1, array2):
    '''
    '''

    return np.sqrt(np.sum(np.square(array1 - array2), axis=1))


def compute_l0_norm(array):
    '''
    '''
    result_list = []
    for i in range(array.shape[0]):
        result_list.append(len(np.where(array[i]!=0)[0]))
    return np.array(result_list)