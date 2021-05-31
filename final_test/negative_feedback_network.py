# -*- encoding: utf-8 -*-
'''
@File    :   negative_feedback_network.py
@Time    :   2021/04/28 14:58:13
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from collections import defaultdict
from tabulate import tabulate


class Negative_feedback_network():
    def __init__(self, x, y, w, lr, c=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
        self.lr = lr
        self.c = c if c else None

    def run(self, iteration, log=False, stable=False):
        result = defaultdict(list)
        
        for i in range(iteration):
            if not stable:
                e = self.x - np.dot(np.transpose(self.w), self.y)
                We = np.dot(self.w, e)
                self.y += self.lr * We
            
            if stable:
                Wy = np.maximum(np.dot(np.transpose(self.w), self.y), self.c[1])
                e = self.x / Wy
                W_ = self.w / np.sum(self.w, axis=1)[:, None]
                We = np.dot(W_, e)
                self.y = np.maximum(self.y, self.c[0]) * We
            
            if log:
                result["e"].append(np.transpose(e))
                result["We"].append(np.transpose(We))
                result["y"].append(np.transpose(self.y.copy()))
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
      
if __name__ == "__main__":
    x = [[1], [1], [0]]
    w = [[1, 1, 0], [1, 1, 1]]
    y = [[0.0], [0.0]]
    lr = 0.25

    # model = Negative_feedback_network(x=x, y=y, w=w, lr=lr)
    # model.run(iteration=5, log=True)

    c = [0.01, 0.01]
    model = Negative_feedback_network(x=x, y=y, w=w, lr=lr, c=c)
    model.run(iteration=5, log=True, stable=True)