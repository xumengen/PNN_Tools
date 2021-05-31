# -*- encoding: utf-8 -*-
'''
@File    :   pseudoinverse.py
@Time    :   2021/04/27 14:52:22
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate


class Pseudoinverse(ModelBase):
    def __init__(self, x, y, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = np.array(b)

    def train(self, log=False):
        '''
        Ya = b
        a = Y'b
        '''
        result = defaultdict(list)
        Y = []
        for i in range(len(self.x)):
            if self.y[i] == 1:
                input_x = np.transpose(np.insert(self.x[i], 0, 1))
            else:
                input_x = -np.transpose(np.insert(self.x[i], 0, 1))
            Y.append(input_x)
        Y = np.array(Y)
        Y_pinv = np.linalg.pinv(Y)
        a = Y_pinv.dot(self.b)
        if log:
            result["a"].append(a)
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))



if __name__ == "__main__":
    x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    y = [1, 1, 1, 2, 2, 2]
    b1 = [1, 1, 1, 1, 1, 1]
    b2 = [2, 2, 2, 1, 1, 1]
    b3 = [1, 1, 1, 2, 2, 2]

    model = Pseudoinverse(x=x, y=y, b=b)
    model.train(log=True)