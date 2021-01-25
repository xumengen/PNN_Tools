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


class Widrow_Hoff(ModelBase):
    def __init__(self, x, y, label, a, b, lr):
        super(Widrow_Hoff, self).__init__()
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = label
        self.a = np.array(a)
        self.b = np.array(b)
        self.lr = lr

    def train(self, epoch):
        for i in range(epoch):
            for j in range(len(self.x)):
                if self.y[j] in self.label:
                    input_x = np.transpose(np.insert(self.x[j], 0, 1))
                else:
                    input_x = np.transpose(np.insert(-self.x[j], 0, -1))
                g_value = np.dot(self.a, input_x)
                print("parameter g of {}-th iteration is {}\n".format(i*len(self.x)+j+1, g_value))
                if g_value != self.b[j]:
                    self.a = self.update(j, g_value, input_x)
                print("parameter a of {}-th iteration is {}\n".format(i*len(self.x)+j+1, self.a))
        # print("parameter a of {}-th iteration is {}\n".format(i*len(self.x)+j+1, self.a))

    def update(self, j, g_value, input_x):
        a_new = self.a + self.lr * (self.b[j] - g_value) * np.transpose(input_x)
        return a_new

    def test(self):
        count = 0
        for i in range(len(self.x)):
            input_x = np.transpose(np.insert(self.x[i], 0, 1))
            g_value = np.dot(self.a, input_x)
            if g_value > 0:
                if self.y[i] == 0:
                    count += 1
            else:
                if self.y[i] == 1 or self.y[i] == 2:
                    count += 1
        print("the percentage is {}\n".format(count/len(self.x)))



if __name__ == "__main__":
    # x = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
    # y = [1, 1, 1, -1, -1]
    # a = [-1.5, 5, -1]
    # b = [2, 2, 2, 2, 2]
    # lr = 0.2

    # x = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]
    # y = [1, 1, 1, -1, -1, -1]
    # label = [1] 
    # a = [1.0, 0.0, 0.0]
    # b = [1.0, 0.5, 1.5, 2.5, 1.5, 1.0]
    # lr = 0.1

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    label = [0]
    a = [0.5, -1.5, 1.5, 2.5, -2.5]
    b = [1 for _ in range(len(x))]
    lr = 0.01

    model = Widrow_Hoff(x=x, y=y, label=label, a=a, b=b, lr=lr)
    model.test()
    model.train(epoch=2)
    model.test()

    ## KNN
    # neigh = KNeighborsClassifier(n_neighbors=1)
    # neigh.fit(x, y)
    # print(neigh.predict([[7.2, 3.5, 4.0, 2.4],
    #                      [7.7, 3.1, 5.8, 1.9],
    #                      [4.7, 2.8, 4.8, 1.0],
    #                      [6.2, 2.4, 1.4, 0.9],
    #                      [7.7, 3.9, 5.4, 2.0]]))
    

