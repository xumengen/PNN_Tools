# -*- encoding: utf-8 -*-
'''
@File    :   batch_delta_learning.py
@Time    :   2021/04/28 13:47:57
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Batch_delta_learning(ModelBase):
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
            gt_result = []
            for j in range(len(self.x)):
                input_x = np.transpose(np.insert(self.x[j], 0, 1))
                output = np.dot(self.w, input_x)
                if output > 0:
                    output = 1
                elif output < 0:
                    output = 0
                else:
                    output = 0.5
                output_result.append(output)
                input_x_result.append(input_x)
                gt_result.append(self.y[j])
            self.update(output_result, input_x_result)
            if log:
                result["x"].append(input_x_result)
                result["t(gt_value)"].append(gt_result)
                result["y(pred_value)"].append(output_result)
                result["w"].append(self.w.copy())
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output_result, input_x_result):
        update_result = np.array([0, 0])
        for idx, result in enumerate(output_result):
            update_result += self.lr * (self.y[idx] - result) * input_x_result[idx]
        self.w += update_result


if __name__ == "__main__":
    x = [[0], [1]]
    y = [1, 0]
    w =  [-1.5, 2]
    lr = 1

    model = Batch_delta_learning(x=x, y=y, w=w, lr=lr)
    model.train(epoch=10, log=True)