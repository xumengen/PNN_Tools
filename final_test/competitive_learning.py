# -*- encoding: utf-8 -*-
'''
@File    :   competitive_learning.py
@Time    :   2021/05/13 11:32:11
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate
from metrics import compute_eucli_diff


class Competitive_learning(ModelBase):
    def __init__(self, data, centers, k, sample_order, lr):
        self.data = np.array(data)
        self.centers = np.array(centers)
        self.k = k
        self.sample_order = sample_order
        self.lr = lr

    def train(self, log=False):
        result = defaultdict(list)
        for i in range(len(self.sample_order)):
            dist_result = []
            for j in range(self.k):
                dist_result.append(compute_eucli_diff(self.data[self.sample_order[i]-1], self.centers[j]))
            min_index = dist_result.index(min(dist_result))
            self.update(min_index, self.sample_order[i]-1)

            if log:
                result['sample x'].append(self.data[self.sample_order[i]-1])
                result['centers'].append(self.centers.copy())
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))
            
    def update(self, min_index, i):
        self.centers[min_index] += self.lr * (self.data[i] - self.centers[min_index])
        
        


if __name__ == "__main__":
    data = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    centers = [[-0.5, 1.5], [0, 2.5], [1.5, 0]]
    k = 3
    lr = 0.1
    sample_order = [3, 1, 1, 5, 6]

    model = Competitive_learning(data=data, centers=centers, k=k, lr=lr, sample_order=sample_order)
    model.train(log=True)