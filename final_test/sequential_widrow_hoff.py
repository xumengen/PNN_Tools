# -*- encoding: utf-8 -*-
'''
@File    :   widrow_hoff.py
@Time    :   2021/04/27 20:11:11
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Sequential_widrow_hoff(ModelBase):
    def __init__(self, x, y, w, lr, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
        self.b = np.array(b)
        self.lr = lr

    def train(self, epoch, log=False):
        result = defaultdict(list)
        for i in range(epoch):
            for j in range(len(self.x)):
                if self.y[j] == 1:
                    input_x = np.transpose(np.insert(self.x[j], 0, 1))
                elif self.y[j] == 2:
                    input_x = -np.transpose(np.insert(self.x[j], 0, 1))
                output = np.dot(self.w, input_x)
                self.update(output, input_x, j)
                if log:
                    result["y"].append(input_x)
                    result["g(x)=ay"].append(output)
                    result["a"].append(self.w[:])
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output, input_x, j):
        self.w = self.w + self.lr * (self.b[j] - output) * input_x



if __name__ == "__main__":
    x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    y = [1, 1, 1, 2, 2, 2]
    w =  [1, 0, 0]
    lr = 0.1
    b = [1, 1, 1, 1, 1, 1]

    model = Sequential_widrow_hoff(x=x, y=y, w=w, lr=lr, b=b)
    model.train(epoch=2, log=True)