# -*- encoding: utf-8 -*-
'''
@File    :   sequential_delta_learning.py
@Time    :   2021/04/28 13:35:52
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Sequential_delta_learning(ModelBase):
    def __init__(self, x, y, w, lr):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
        self.lr = lr

    def train(self, epoch, log=False):
        result = defaultdict(list)
        for i in range(epoch):
            for j in range(len(self.x)):
                input_x = np.transpose(np.insert(self.x[j], 0, 1))
                output = np.dot(self.w, input_x)
                if output > 0:
                    output = 1
                elif output < 0:
                    output = 0
                else:
                    output = 0.5
                self.update(output, j, input_x)
                if log:
                    result["x"].append(input_x)
                    result["t(gt_value)"].append(self.y[j])
                    result["y(pred_value)"].append(output)
                    result["w"].append(self.w[:])
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output, j, input_x):
        self.w = self.w + self.lr * (self.y[j] - output) * input_x


if __name__ == "__main__":
    x = [[0], [1]]
    y = [1, 0]
    w =  [-1.5, 2]
    lr = 1

    model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    model.train(epoch=10, log=True)