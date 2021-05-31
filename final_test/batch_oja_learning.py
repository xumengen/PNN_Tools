# -*- encoding: utf-8 -*-
'''
@File    :   batch_oja_learning.py
@Time    :   2021/04/29 17:08:18
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Batch_oja_learning(ModelBase):
    def __init__(self, x, w, lr):
        self.x = np.array(x)
        self.w = np.array(w)
        self.lr = lr

    def train(self, epoch, log=False):
        mean_array = np.mean(self.x, axis=0)
        self.x -= mean_array
        result = defaultdict(list)
        for i in range(epoch):
            x_result = []
            output_y_result = []
            output_result = []
            for j in range(len(self.x)):
                y = np.dot(self.w, np.transpose(self.x[j]))
                output = self.lr * y * (self.x[j] - y * self.w)
                x_result.append(self.x[j])
                output_y_result.append(y)
                output_result.append(output)
            self.update(output_result)
            if log:
                result["wx"].append(output_y_result)
                result["w"].append(self.w.copy())
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output_result):
        self.w += np.sum(np.array(output_result), axis=0)


if __name__ == "__main__":
    x = np.array([[0.0, 1.0],
                     [3.0, 5.0],
                     [5.0, 4.0],
                     [5.0, 6.0],
                     [8.0, 7.0],
                     [9.0, 7.0]])
    w =  [-1.0, 0.0]
    lr = 0.01

    model = Batch_oja_learning(x=x, w=w, lr=lr)
    model.train(epoch=6, log=True)