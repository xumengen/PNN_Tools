# -*- encoding: utf-8 -*-
'''
@File    :   convolutional.py
@Time    :   2021/04/28 22:28:15
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import torch
import torch.nn.functional as F
import numpy as np

if __name__ == "__main__":
    input_data = [[[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]],
                  [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],]

    mask = [[[1, -0.1], [1, -0.1]],
            [[0.5, 0.5], [-0.5, -0.5]],]

    input_data_tensor = torch.tensor(np.expand_dims(input_data, axis=0))
    mask_tensor = torch.tensor(np.expand_dims(mask, axis=0))
    result = F.conv2d(input_data_tensor, mask_tensor, stride=1, padding=0)
    print(result)