import os
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    pca = PCA(n_components=2)
    pca.fit(x)
    new_x = np.array([[7.6, 4.2, 6.4, 1.7],
                      [4.9, 2.9, 2.6, 2.3],
                      [4.3, 2.8, 4.1, 1.7],
                      [6.1, 2.6, 1.9, 0.2],
                      [6.8, 2.4, 1.1, 1.9]])
    print(pca.transform(new_x))