# -*- encoding: utf-8 -*-
'''
@File    :   fisher_method.py
@Time    :   2021/04/30 21:39:53
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def compute_cost(data_array, class_array, w):

    data_array_1 = data_array[np.where(class_array==1)]
    data_array_2 = data_array[np.where(class_array==2)]

    mean_array_1 = np.mean(data_array_1, axis=0)
    mean_array_2 = np.mean(data_array_2, axis=0)

    # between class
    sb = np.square(np.dot(w, np.transpose(mean_array_1 - mean_array_2)))

    # within class
    sw = np.sum(np.square(np.dot(w, (data_array_1 - mean_array_1).T))) + \
        np.sum(np.square(np.dot(w, (data_array_2 - mean_array_2).T)))

    return sb/sw

def Fisher_method(data_array, class_array, w_array):

    cost_result = []
    for w in w_array:
        cost_result.append(compute_cost(data_array, class_array, w))
    
    print("the cost results of array w are:\n", cost_result)

    more_effective_w = w_array[np.argsort(np.array(cost_result))[::-1][0]]
    print("the more effective weight is:\n", more_effective_w)


if __name__ == "__main__":
    data_array = np.array([[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]])
    class_array = np.array([1, 1, 1, 2, 2])
    w_array = np.array([[-1, 5], [2, -3]])

    Fisher_method(data_array=data_array, class_array=class_array, w_array=w_array)