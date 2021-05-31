# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2021/05/12 17:38:35
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def compute_SAD_diff(array1, array2):
    """
    """
    return np.sum(np.absolute(array1 - array2))


def compute_eucli_diff(array1, array2):
    """
    """
    return np.sqrt(np.sum(np.square(array1 - array2)))

def compute_l2_diff(array1, array2):
    """
    """
    return np.sum(np.square(array1 - array2))