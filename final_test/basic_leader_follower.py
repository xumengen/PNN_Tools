# -*- encoding: utf-8 -*-
'''
@File    :   basic_leader_follower.py
@Time    :   2021/05/13 14:33:56
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate
from metrics import compute_eucli_diff


class Basic_leader_follower(ModelBase):
    def __init__(self, data, lr, thres, sample_order):
        self.data = np.array(data)
        self.thres = thres
        self.sample_order = sample_order
        self.lr = lr
        self.cluster_result = [self.data[self.sample_order[0]-1], ]

    def train(self, log=False):
        result = defaultdict(list)
        for i in range(len(self.sample_order)):
            dist_result = []
            for cluster in self.cluster_result:
                dist_result.append(compute_eucli_diff(self.data[self.sample_order[i]-1], cluster))
            min_index = dist_result.index(min(dist_result))
            if min(dist_result) < self.thres:
                self.update(min_index, self.sample_order[i]-1)
            else:
                self.cluster_result.append(self.data[self.sample_order[i]-1])

            if log:
                result['sample x'].append(self.data[self.sample_order[i]-1])
                result['centers'].append(self.cluster_result.copy())

        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))

    def update(self, min_index, i):
        self.cluster_result[min_index] = self.cluster_result[min_index] + self.lr * (self.data[i] - self.cluster_result[min_index])
        


if __name__ == "__main__":
    data = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    lr = 0.5
    thres = 3
    sample_order = [3, 1, 1, 5, 6]

    model = Basic_leader_follower(data=data, lr=lr, thres=thres, sample_order=sample_order)
    model.train(log=True)