# -*- encoding: utf-8 -*-
'''
@File    :   batch_normalization.py
@Time    :   2021/04/28 18:25:29
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np

def Batch_normalization(data, beta, gama, epsilon):

    mean_array = np.mean(data, axis=0)
    var_array = np.var(data, axis=0)

    data = beta + gama * (data - mean_array) / np.sqrt(var_array + epsilon)

    return data

if __name__ == '__main__':
    data = [ [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],
             [[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]],
             [[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]],
             [[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]],]

    print("Batch normalization:", Batch_normalization(data, beta=0, gama=1, epsilon=0.1))