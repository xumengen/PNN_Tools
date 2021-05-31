# -*- encoding: utf-8 -*-
'''
@File    :   sparse_code.py
@Time    :   2021/04/30 22:39:57
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def select_sparser_code(sparse_code_array, dictionary, x):

    error_result = []
    for sparse_code in sparse_code_array:
        error = np.sqrt(np.sum(np.square(x.T - np.dot(dictionary, np.transpose(sparse_code)))))
        error_result.append(error)
    print("the reconstruction error results are:\n", error_result)

    lowest_error_result = error_result[np.argsort(error_result)[0]]
    print("the better solution is {} and the error is {}".format(np.argsort(error_result)[0]+1, lowest_error_result))


def select_sparser_codev2(sparse_code_array, dictionary, x, lam):

    error_result = []
    for sparse_code in sparse_code_array:
        error = np.sqrt(np.sum(np.square(x.T - np.dot(dictionary, np.transpose(sparse_code)))))
        error += np.linalg.norm(sparse_code, ord=0)
        error_result.append(error)
    print("the reconstruction error results are:\n", error_result)

    lowest_error_result = error_result[np.argsort(error_result)[0]]
    print("the better solution is {} and the error is {}".format(np.argsort(error_result)[0]+1, lowest_error_result))

if __name__ == "__main__":
    sparse_code_array = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, -1, 0]])
    dictionary = np.array([[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
                           [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]])
    x = np.array([-0.05, -0.95])

    select_sparser_code(sparse_code_array=sparse_code_array, dictionary=dictionary, x=x)
