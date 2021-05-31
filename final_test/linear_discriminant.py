# -*- encoding: utf-8 -*-
'''
@File    :   linear_discriminant.py
@Time    :   2021/05/29 13:34:22
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np

def linear_discriminant(w_array, input_x):
    return np.dot(w_array, input_x.T)

if __name__ == "__main__":
    w_array = np.array([[1, 0.5, 0.5],
               [-1, 2, 2],
               [2, -1, -1]])
    input_x = np.array([[1, 0, 1]])
    print(linear_discriminant(w_array=w_array, input_x=input_x))