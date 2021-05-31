# -*- encoding: utf-8 -*-
'''
@File    :   activation.py
@Time    :   2021/04/28 18:05:04
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def ReLU(data):
    return np.where(data>=0, data, 0)

def LReLU(data, a):
    return np.where(data>=0, data, a*data)

def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))

def Heaviside(data, thres):
    data -= thres
    data[data>0] = 1
    data[data==0] = 0.5
    data[data<0] = 0
    return data

def symmetric_sigmoid(data):
    return 2 / (1 + np.exp(-2*data)) - 1

def logarithmic_sigmoid(data):
    return 1 / (1 + np.exp(-data))

def radial_bias(data):
    return np.exp(-np.square(data))