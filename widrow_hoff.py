# -*- encoding: utf-8 -*-
'''
@File    :   widrow_hoff.py
@Time    :   2021/01/23 13:37:16
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np
from base import ModelBase
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from tabulate import tabulate


class Widrow_Hoff(ModelBase):
    def __init__(self, x, y, a, b, lr):
        super(Widrow_Hoff, self).__init__()
        self.x = np.array(x)
        self.y = np.array(y)
        self.a = np.array(a)
        self.b = np.array(b)
        self.lr = lr

    def train(self, epoch, log=False):
        result = defaultdict(list)
        for i in range(epoch):
            for j in range(len(self.x)):
                if self.y[j] == 1:
                    input_x = np.transpose(np.insert(self.x[j], 0, 1))
                else:
                    input_x = np.transpose(np.insert(-self.x[j], 0, -1))
                g_value = np.dot(self.a, input_x)
                if g_value != self.b[j]:
                    self.a = self.update(j, g_value, input_x)
                if log:
                    result["pred_value"].append(g_value)
                    result["parameter"].append(self.a[:])
        if log:
            print(tabulate(result, headers='keys', showindex='always', tablefmt='pretty'))

    def update(self, j, g_value, input_x):
        a_new = self.a + self.lr * (self.b[j] - g_value) * np.transpose(input_x)
        return a_new

    def test(self):
        count = 0
        for i in range(len(self.x)):
            input_x = np.transpose(np.insert(self.x[i], 0, 1))
            g_value = np.dot(self.a, input_x)
            if g_value >= 0 and self.y[i] == 1:
                count += 1
            elif g_value < 0 and self.y[i] == -1:
                count += 1
        print("the percentage is {}\n".format(count/len(self.x)))


def change_label(input):
    res = list()
    for idx, label in enumerate(input):
        if label == 0:
            res.append(1)
        else:
            res.append(-1)
    return res


if __name__ == "__main__":
    # x = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
    # y = [1, 1, 1, -1, -1]
    # a = [-1.5, 5, -1]
    # b = [2, 2, 2, 2, 2]
    # lr = 0.2

    # x = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]
    # y = [1, 1, 1, -1, -1, -1]
    # a = [1.0, 0.0, 0.0]
    # b = [1.0, 0.5, 1.5, 2.5, 1.5, 1.0]
    # lr = 0.1

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    y = change_label(y)
    a = [0.5, -1.5, 2.5, -0.5, -2.5]
    b = [1 for _ in range(len(x))]
    lr = 0.01

    model = Widrow_Hoff(x=x, y=y, a=a, b=b, lr=lr)
    model.test()
    model.train(epoch=2, log=False)
    model.test()

    ## KNN
    # neigh = KNeighborsClassifier(n_neighbors=1)
    # neigh.fit(x, y)
    # print(neigh.predict([[7.2, 3.5, 4.0, 2.4],
    #                      [7.7, 3.1, 5.8, 1.9],
    #                      [4.7, 2.8, 4.8, 1.0],
    #                      [6.2, 2.4, 1.4, 0.9],
    #                      [7.7, 3.9, 5.4, 2.0]]))
    

