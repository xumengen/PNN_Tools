# -*- encoding: utf-8 -*-
'''
@File    :   compute_output_size.py
@Time    :   2021/04/28 23:39:53
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np


def compute_conv_output_size(input_size, mask_size, stride, padding):
    
    ''' 
    input_size: H1*W1*C*N1
    mask_size: H2*W2*C*N2
    '''

    output_height = 1 + (input_size[0] - mask_size[0] + 2 * padding) / stride
    output_width = 1 + (input_size[1] - mask_size[1] + 2 * padding) / stride

    return [output_height, output_width, mask_size[-1], input_size[-1]]

def compute_pool_output_size(input_size, mask_size, stride):

    '''
    input_size: H1*W1*C*N1
    mask_size: H2*W2
    '''

    output_height = 1 + (input_size[0] - mask_size[0]) / stride
    output_width = 1 + (input_size[1] - mask_size[1]) / stride

    return [output_height, output_width, input_size[-2], input_size[-1]]