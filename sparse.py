import os
import numpy as np
from sklearn import datasets
from sklearn.decomposition import sparse_encode

from metrics import compute_l2_norm, compute_l0_norm

if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    class_0 = x[np.where(y==0)]
    class_1 = x[np.where(y==1)]
    class_2 = x[np.where(y==2)]
    # print(class_0.shape)
    # print(class_1.shape)
    # print(class_2.shape)

    new_x = np.array([[7.8, 3.5, 6.8, 0.7],
                      [6.6, 3.9, 6.0, 1.5],
                      [4.7, 3.4, 1.0, 1.9],
                      [5.5, 2.6, 3.6, 2.0],
                      [5.1, 3.0, 5.1, 2.2]])

    numNonZero=2
    tolerance=1.000000e-05
    y_0 = sparse_encode(new_x, class_0, algorithm='omp', n_nonzero_coefs=numNonZero, alpha=tolerance)
    y_1 = sparse_encode(new_x, class_1, algorithm='omp', n_nonzero_coefs=numNonZero, alpha=tolerance)
    y_2 = sparse_encode(new_x, class_2, algorithm='omp', n_nonzero_coefs=numNonZero, alpha=tolerance)
    # print(y_0.shape)
    # print(y_1.shape)
    # print(y_2.shape)

    recon_x_0 = y_0.dot(class_0)
    recon_x_1 = y_1.dot(class_1)
    recon_x_2 = y_2.dot(class_2)
    # print(recon_x_0.shape)
    # print(recon_x_1.shape)
    # print(recon_x_2.shape)

    cost_0 = compute_l2_norm(new_x, recon_x_0) + 0.1 * compute_l0_norm(y_0)
    cost_1 = compute_l2_norm(new_x, recon_x_1) + 0.1 * compute_l0_norm(y_1)
    cost_2 = compute_l2_norm(new_x, recon_x_2) + 0.1 * compute_l0_norm(y_2)
    print(cost_0)
    print(cost_1)
    print(cost_2)
   
