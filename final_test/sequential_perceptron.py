# -*- encoding: utf-8 -*-
'''
@File    :   sequential_perceptron.py
@Time    :   2021/04/26 18:37:58
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Sequential_perceptron(ModelBase):
    def __init__(self, x, y, w, lr):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
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
                self.update(output, input_x)
                if log:
                    result["y"].append(input_x)
                    result["g(x)=ay"].append(output)
                    result["parameter"].append(self.w[:])
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output, input_x):
        if output <= 0:
            self.w = self.w + self.lr * input_x


if __name__ == "__main__":
    x = [[1, 5], [2, 5], [4, 1], [5, 1]]
    y = [1, 1, 2, 2]
    w =  [-25, 6, 3]
    lr = 1

    model = Sequential_perceptron(x=x, y=y, w=w, lr=lr)
    model.train(epoch=2, log=True)