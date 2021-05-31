# -*- encoding: utf-8 -*-
'''
@File    :   batch_perceptron.py
@Time    :   2021/04/26 18:25:05
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Batch_perceptron(ModelBase):
    def __init__(self, x, y, w, lr):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
        self.lr = lr

    def train(self, epoch, log=False):
        result = defaultdict(list)
        for i in range(epoch):
            output_result = []
            input_x_result = []
            for j in range(len(self.x)):
                if self.y[j] == 1:
                    input_x = np.transpose(np.insert(self.x[j], 0, 1))
                elif self.y[j] == 2:
                    input_x = -np.transpose(np.insert(self.x[j], 0, 1))
                input_x_result.append(input_x)
                output = np.dot(self.w, input_x)
                output_result.append(output)
            self.update(output_result, input_x_result)
            symbol = self.epoch_end(np.array(output_result))
            if log:
                result["y(input_x_result)"].append(input_x_result)
                result["g(x)(pred_value)"].append(output_result)
                result["w(parameter)"].append(self.w[:])
                result["converged"].append(symbol)
            
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
            
      
    def update(self, output_result, input_x_result):
        for idx, result in enumerate(output_result):
            if result <= 0:
                self.w = self.w + self.lr * input_x_result[idx]

    def epoch_end(self, output_result):
        return True if (output_result > 0).all() else False



if __name__ == "__main__":
    x = [[1, 5], [2, 5], [4, 1], [5, 1]]
    y = [1, 1, 2, 2]
    w =  [-25, 6, 3]
    lr = 1

    model = Batch_perceptron(x=x, y=y, w=w, lr=lr)
    model.train(epoch=3, log=True)