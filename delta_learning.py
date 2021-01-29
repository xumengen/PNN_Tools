# -*- encoding: utf-8 -*-
'''
@File    :   delta_learning.py
@Time    :   2021/01/29 20:29:11
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np
from base import ModelBase
from sklearn import datasets
from collections import defaultdict
from tabulate import tabulate


class Delta_Learning(ModelBase):
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
                    result["pred_value"].append(output)
                    result["parameter"].append(self.w[:])
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
    def update(self, output, j, input_x):
        self.w = self.w + self.lr * (self.y[j] - output) * input_x

    def test(self):
        count = 0
        for i in range(len(self.x)):
            input_x = np.transpose(np.insert(self.x[i], 0, 1))
            output = np.dot(self.w, input_x)
            if output > 0:
                output = 1
            elif output < 0:
                output = 0
            else:
                output = 0.5
            if output == self.y[i]:
                count += 1
        print("the percentage is {}%\n".format(count/len(self.x)*100))


def change_label(input):
    res = list()
    for idx, label in enumerate(input):
        if label == 0:
            res.append(1)
        else:
            res.append(0)
    return res


if __name__ == "__main__":
    # x = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
    # y = [1, 1, 1, 0, 0]
    # w =  [-1.5, 5, -1]
    # lr = 1

    # x = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]
    # y = [1, 1, 1, 0, 0, 0]
    # w = [0.5, -5.5, -0.5]
    # lr = 1

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    y = change_label(y)
    w = [-0.5, 3.5, -2.5, -2.5, 0.5]
    # w = [0.5, 2.5, -0.5, -3.5, 3.5]
    lr = 0.10

    model = Delta_Learning(x=x, y=y, w=w, lr=lr)
    model.test()
    model.train(epoch=2, log=True)
    model.test()