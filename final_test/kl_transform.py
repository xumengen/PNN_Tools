# -*- encoding: utf-8 -*-
'''
@File    :   KL_transform.py
@Time    :   2021/04/29 15:21:12
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def KL_transform(data, dimension):

    mean_array = np.mean(data, axis=1)

    covariance_matrix = np.cov(data, bias=True)

    print("The covariance matrix is:\n", covariance_matrix)

    E, V = np.linalg.eig(covariance_matrix)
    
    print("The eigenvectors is:\n", V)
    print("The eigenvalues is:\n", E)
    
    topd_index = E.argsort()[::-1][:dimension]
    V_topd = V[topd_index]

    return np.dot(V_topd, data-mean_array.T[:, None])

if __name__ == "__main__":
    data = np.array([[1, 2, 1],
            [2, 3, 1],
            [3, 5, 1],
            [2, 2, 1]])

    # data = np.array([[0, 1],
    #                  [3, 5],
    #                  [5, 4],
    #                  [5, 6],
    #                  [8, 7],
    #                  [9, 7]])

    new_data = KL_transform(data.T, dimension=2)

    
