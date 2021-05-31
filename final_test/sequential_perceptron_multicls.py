# -*- encoding: utf-8 -*-
'''
@File    :   sequential_perceptron_multicls.py
@Time    :   2021/04/26 22:23:30
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Sequential_perceptron_multicls(ModelBase):
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
                output_result = []
                for m in range(len(self.w)):
                    output_result.append(np.dot(self.w[m], input_x))
                self.update(output_result, input_x, j)
                if log:
                    result["y"].append(input_x)
                    result["g(x)=ay"].append(output_result)
                    result["parameter"].append(self.w.copy())
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output_result, input_x, j):
        max_result = -float('inf')
        predict_class = None
        for idx, result in enumerate(output_result):
            if result >= max_result:
                max_result = result
                predict_class = idx + 1
        gt_class = self.y[j]

        if predict_class != gt_class:
            self.w[gt_class-1] = self.w[gt_class-1] + self.lr * input_x
            self.w[predict_class-1] = self.w[predict_class-1] - self.lr * input_x



if __name__ == "__main__":
    x = [[1, 1], [2, 0], [0, 2], [-1, 1], [-1, -1]]
    y = [1, 1, 2, 2, 3]
    w =  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    lr = 1

    model = Sequential_perceptron_multicls(x=x, y=y, w=w, lr=lr)
    model.train(epoch=2, log=True)