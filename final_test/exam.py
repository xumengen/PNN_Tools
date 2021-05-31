# -*- encoding: utf-8 -*-
'''
@File    :   exam.py
@Time    :   2021/05/29 12:55:08
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from linear_discriminant import linear_discriminant
from sequential_perceptron_multicls import Sequential_perceptron_multicls
from negative_feedback_network import Negative_feedback_network
from sparse_code import select_sparser_code, select_sparser_codev2
from activation import Heaviside
from sequential_delta_learning import Sequential_delta_learning
from sklearn.svm import SVC
from sequential_perceptron import Sequential_perceptron
from radial_basis_function import Radial_basis_function
from batch_oja_learning import Batch_oja_learning
from k_means import k_means
from sklearn.cluster import AgglomerativeClustering
from agglomerative_clustering import agglomerative_hierarchical_clusteringv2
from competitive_learning import Competitive_learning
from fisher_method import Fisher_method


if __name__ == "__main__":

    # # 2019 1.e
    # X = [[5, 1], [5, -1], [3, 0], [2, 1], [4, 2]]
    # Y = [1, 1, 2, 2, 2]

    # neigh_1 = KNeighborsClassifier(n_neighbors=1)
    # neigh_1.fit(X, Y)
    # print(neigh_1.predict([[4, 0]]))

    # neigh_3 = KNeighborsClassifier(n_neighbors=3)
    # neigh_3.fit(X, Y)
    # print(neigh_3.predict([[4, 0]]))
    
    # # 2019 2.b
    # w_array = np.array([[1, 0.5, 0.5],
    #            [-1, 2, 2],
    #            [2, -1, -1]])
    # input_x = np.array([[1, 0, 1]])
    # print(linear_discriminant(w_array=w_array, input_x=input_x))

    # # 2019 2.e
    # x = [[0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.25], [1, 1, 1], [0, 0, 0]]
    # y = [1, 1, 2, 2, 3]
    # w =  [[1, 0.5, 0.5, -0.75], [-1, 2, 2, 1], [2, -1, -1, 1]]
    # lr = 1

    # model = Sequential_perceptron_multicls(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2019 3.a
    # w = np.array([[-1, 3]])
    # x = np.array([[2, 0.5]])
    # print(Heaviside(np.dot(w, x.T), thres=-2))

    # # 2019 3.d
    # x = [[1], [1], [0]]
    # w = [[1, 1, 0], [1, 1, 1]]
    # y = [[0.0], [0.0]]
    # lr = 0.25

    # model = Negative_feedback_network(x=x, y=y, w=w, lr=lr)
    # model.run(iteration=4, log=True)

    # # 2019 3.f
    # sparse_code_array = np.array([[1, 2, 0, -1],
    #                               [0, 0.5, 1, 0]])
    # dictionary = np.array([[1, 1, 2, 1],
    #                        [-4, 3, 2, -1]])
    # x = np.array([2, 3])

    # select_sparser_codev2(sparse_code_array=sparse_code_array, dictionary=dictionary, x=x, lam=1)

    # # 2018 2.d.ii
    # w = np.array([[3, 0.5]])
    # input_x = np.array([[2, -1],
    #                     [-1, 0],
    #                     [0, 0],
    #                     [1, 1],
    #                     [0, -1]])
    # print(Heaviside(np.dot(w, input_x.T), thres=-1))

    # # 2018 2.d.iii
    # x = np.array([[2, -1],
    #               [-1, 0],
    #               [0, 0],
    #               [1, 1],
    #               [0, -1]])
    # y = [0, 1, 1, 0, 1]
    # w =  [1, 3, 0.5]
    # lr = 1

    # model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2018 4.d
    # # Z = W1W2X
    # Z = np.array([[98, 7.5],
    #               [-168, -246]])
    # W2 = np.array([[8, -4],
    #                [6, 9]])
    # X = np.array([[2, -0.5],
    #               [-4, -6]])
    # W1 = np.dot(Z, np.linalg.pinv(np.dot(W2, X)))
    # print("W1: {}".format(W1))

    #TODO 2018 5.d.i Adaboost

    # #2018 6.b.ii
    # X = np.array([[-2, 4], [-1, 1], [2, 4]])
    # Y = np.array([1, -1, -1])
    # clf = SVC(kernel='linear')
    # clf.fit(X, Y)
    # print("w and b of svm classifier is {}, {}".format(clf.coef_, clf.intercept_))

    # # 2017 2.d
    # x = [[5, 1], [5, -1], [7, 0], [3, 0], [2, 1], [1, -1]]
    # y = [1, 1, 1, 2, 2, 2]
    # w =  [-25, 5, 2]
    # lr = 1

    # model = Sequential_perceptron(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2017 2.e.ii
    # X = np.array([[5, 1], [5, -1], [7, 0], [3, 0], [2, 1], [1, -1]])
    # Y = np.array([1, 1, 1, 2, 2, 2])
    # clf = SVC(kernel='linear')
    # clf.fit(X, Y)
    # print("w and b of svm classifier is {}, {}".format(clf.coef_, clf.intercept_))

    # # 2017 3.f w0 = -theta
    # x = [[0, 2], [2, 1], [-3, 1], [-2, -1], [0, -1]]
    # y = [1, 1, 0, 0, 0]
    # w =  [2, 0.5, 1]
    # lr = 1

    # model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2017 4.b
    # input_data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # target_data = np.array([[-1, 1, 1, -1]])
    # centers = np.array([[-1, -1], [1, 1]])
    # stds = np.array([np.sqrt(2), np.sqrt(2)])
    # net = Radial_basis_function(input_data, target_data, centers, stds)

    # # 2017 5.c.i
    # X = np.array([[1, 2], [3, 5], [5, 4], [8, 7], [11, 7]])
    # print(X-np.mean(X, axis=0))

    # # 2017 5.c.ii
    # x = np.array([[-4.6, -3],
    #               [-2.6, 0],
    #               [-0.6, -1],
    #               [2.4, 2],
    #               [5.4, 2]])
    # w =  [-1.0, 0.0]
    # lr = 0.01

    # model = Batch_oja_learning(x=x, w=w, lr=lr)
    # model.train(epoch=2, log=True)

    # # 2017 6.b
    # feature_vector_array = [[1, 0], [0, 2], [1, 3], [3, 0], [3, 1]]
    # k = 2
    # ori_feature_vetor_array = [[3, 2], [4, 0]]
    # k_means(feature_vector_array=feature_vector_array, k=2, ori_feature_vetor_array=ori_feature_vetor_array, method='eucli')

    # # 2017 6.c
    # data = [[1, 0], [0, 2], [1, 3], [3, 0], [3, 1]]
    # agglomerative_hierarchical_clusteringv2(data, k=2, method='eucli', cluster_method='single_link')

    # clustering = AgglomerativeClustering(n_clusters=2, linkage='single', compute_distances=True).fit(data)
    # print(clustering.labels_)

    # # 2017 6.d
    # data = [[1, 0], [0, 2], [1, 3], [3, 0], [3, 1]]
    # centers = [[1.0, 1.0], [2.0, 2.0]]
    # k = 2
    # lr = 0.5
    # sample_order = [1, 2, 3, 4, 5]

    # model = Competitive_learning(data=data, centers=centers, k=k, lr=lr, sample_order=sample_order)
    # model.train(log=True)

    #TODO competitive learning algorithm (with normalisation)

    # # 2016 2.c
    # X = [[-2, 6], [-1, -4], [3, -1], [-3, -2], [-4, -5]]
    # Y = [1, 1, 1, 2, 3]
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(X, Y)
    # print(neigh.predict([[-2, 0]]))

    # # 2016 3.b.ii
    # data_array = np.array([[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]])
    # class_array = np.array([1, 1, 1, 2, 2])
    # w_array = np.array([[-1, 5], [2, -6]])

    # Fisher_method(data_array=data_array, class_array=class_array, w_array=w_array)

    # # 2016 6.b
    # X = np.array([[1, 2], [7, 8], [10, 15]])
    # Y = np.array([1, 2, 2])
    # clf = SVC(kernel='linear')
    # clf.fit(X, Y)
    # print("w and b of svm classifier is {}, {}".format(clf.coef_, clf.intercept_))
