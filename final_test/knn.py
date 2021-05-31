# -*- encoding: utf-8 -*-
'''
@File    :   knn.py
@Time    :   2021/05/29 12:47:12
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    X = [[5, 1], [5, -1], [3, 0], [2, 1], [4, 2]]
    Y = [1, 1, 2, 2, 2]

    neigh_1 = KNeighborsClassifier(n_neighbors=1)
    neigh_1.fit(X, Y)
    print(neigh_1.predict([[4, 0]]))

    neigh_3 = KNeighborsClassifier(n_neighbors=3)
    neigh_3.fit(X, Y)
    print(neigh_3.predict([[4, 0]]))
    

    