# -*- encoding: utf-8 -*-
'''
@File    :   tutorial_2.py
@Time    :   2021/04/27 14:31:19
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np

from batch_perceptron import Batch_perceptron
from sequential_perceptron import Sequential_perceptron
from sequential_perceptron_multicls import Sequential_perceptron_multicls
from pseudoinverse import Pseudoinverse
from sequential_widrow_hoff import Sequential_widrow_hoff
from sequential_delta_learning import Sequential_delta_learning
from batch_delta_learning import Batch_delta_learning
from negative_feedback_network import Negative_feedback_network
from activation import ReLU, LReLU, tanh, Heaviside, symmetric_sigmoid, logarithmic_sigmoid, radial_bias
from batch_normalization import Batch_normalization
import torch
import torch.nn.functional as F
from compute_output_size import compute_conv_output_size, compute_pool_output_size
from kl_transform import KL_transform
from batch_oja_learning import Batch_oja_learning
from fisher_method import Fisher_method
from sparse_code import select_sparser_code
from k_means import k_means
from competitive_learning import Competitive_learning
from basic_leader_follower import Basic_leader_follower
from sklearn.cluster import AgglomerativeClustering
from agglomerative_clustering import agglomerative_hierarchical_clusteringv2
import skfuzzy as fuzz
from fuzzy_kmeans import cmeans
from radial_basis_function import Radial_basis_function


if __name__ == "__main__":

    # # tutoriral 2
    # x = [[1, 5], [2, 5], [4, 1], [5, 1]]
    # y = [1, 1, 2, 2]
    # w =  [-25, 6, 3]
    # lr = 1
    
    # # 2.6 class id [1, 2]
    # model = Batch_perceptron(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2.7 class id [1, 2]
    # model = Sequential_perceptron(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=2, log=True)

    # # 2.10 class id [1, 2]
    # x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    # y = [1, 1, 1, 2, 2, 2]
    # w =  [1, 0, 0]
    # lr = 1
    # model = Sequential_perceptron(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 2.11 class id [1, 2, 3, ... , N]
    # x = [[1, 1], [2, 0], [0, 2], [-1, 1], [-1, -1]]
    # y = [1, 1, 2, 2, 3]
    # w =  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # lr = 1

    # model = Sequential_perceptron_multicls(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=2, log=True)

    # # 2.12, 13 class id [1, 2]
    # x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    # y = [1, 1, 1, 2, 2, 2]
    # b1 = [1, 1, 1, 1, 1, 1]
    # b2 = [2, 2, 2, 1, 1, 1]
    # b3 = [1, 1, 1, 2, 2, 2]

    # model = Pseudoinverse(x=x, y=y, b=b3)
    # model.train(log=True)

    # # 2.14 class id [1, 2]
    # x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    # y = [1, 1, 1, 2, 2, 2]
    # w =  [1, 0, 0]
    # lr = 0.1
    # b = [1, 1, 1, 1, 1, 1]

    # model = Sequential_widrow_hoff(x=x, y=y, w=w, lr=lr, b=b)
    # model.train(epoch=2, log=True)

    # # 3.3 class id [1, 0]
    # x = [[0], [1]]
    # y = [1, 0]
    # w =  [-1.5, 2]
    # lr = 1

    # model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=10, log=True)

    # # 3.4 class id [1, 0]
    # model = Batch_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=10, log=True)

    # # 3.5 class id [1, 0]
    # x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # y = [0, 0, 0, 1]
    # w =  [0.5, 1, 1]
    # lr = 1

    # model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=5, log=True)

    # 3.6 class id [1, 0]
    # x = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
    # y = [1, 1, 1, 0, 0, 0]
    # w =  [1, 0, 0]
    # lr = 1

    # model = Sequential_delta_learning(x=x, y=y, w=w, lr=lr)
    # model.train(epoch=3, log=True)

    # # 3.7
    # x = [[1], [1], [0]]
    # w = [[1, 1, 0], [1, 1, 1]]
    # y = [[0.0], [0.0]]
    # lr = 0.5

    # model = Negative_feedback_network(x=x, y=y, w=w, lr=lr)
    # model.run(iteration=5, log=True)

    # c = [0.01, 0.01]
    # model = Negative_feedback_network(x=x, y=y, w=w, lr=lr, c=c)
    # model.run(iteration=5, log=True, stable=True)

    # # 4.4
    # # each row in input_x represents the data point 
    # input_x = np.array([[1, 0, 1, 0], 
    #            [0, 1, 0, 1],
    #            [1, 1, 0, 0]])
    # # each row represents the weights about output node
    # w_1 = np.array([[-0.7057, 1.9061, 2.6605, -1.1359],
    #                 [0.4900, 1.9324, -0.4269, -5.1570],
    #                 [0.9438, -5.4160, -0.3431, -0.2931]])
    # # each column represents bias about output node
    # b_1 = np.array([[4.8432],
    #                 [0.3973],
    #                 [2.1761]])
    # # each column represents the output of each data point
    # y = symmetric_sigmoid(np.dot(w_1, input_x.T) + b_1)
    # print(y)
    
    # w_2 = np.array([[-1.1444, 0.3115, -9.9812],
    #                 [0.0106, 11.5477, 2.6479]])
    # b_2 = np.array([[2.5230],
    #                 [2.6463]])
    # y = logarithmic_sigmoid(np.dot(w_2, y) + b_2)
    # print(y)

    # # 4.6, 4.7
    # input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # target_data = np.array([[0, 1, 1, 0]])
    # centers = np.array([[0, 0], [1, 1]])
    # stds = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # net = Radial_basis_function(input_data, target_data, centers, stds)
    # test_data = np.array([[0.5, -0.1],
    #                       [-0.2, 1.2],
    #                       [0.8, 0.3],
    #                       [1.8, 0.6]])
    # net.pred(test_data=test_data)

    # input_data = np.array([[0.05], [0.2], [0.25], [0.3], [0.4], [0.43], [0.48], [0.6], [0.7], [0.8], [0.9], [0.95]])
    # target_data = np.array([[0.0863, 0.2662, 0.2362, 0.1687, 0.1260, 0.1756, 0.3290, 0.6694, 0.4573, 0.3320, 0.4063, 0.3535]])
    # centers = np.array([[0.2], [0.6], [0.9]])
    # stds = np.array([0.1, 0.1, 0.1])
    # net = Radial_basis_function(input_data, target_data, centers, stds)

    # centers = np.array([[0.1667], [0.35], [0.5525], [0.8833]])
    # stds = np.array([0.7842, 0.7842, 0.7842, 0.7842])
    # net = Radial_basis_function(input_data, target_data, centers, stds)
    # net.pred(test_data=input_data, gt_value=target_data)

    # # 5.4
    # data = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
    # print("ReLU:", ReLU(data))
    # print("LReLU:", LReLU(data, 0.1))
    # print("tanh:", tanh(data))
    # print("Heaviside:", Heaviside(data, 0.1))

    # # 5.5
    # data = [ [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],
    #          [[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]],
    #          [[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]],
    #          [[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]],]

    # print("The result of Batch normalization is:\n", Batch_normalization(data, beta=0, gama=1, epsilon=0.1))

    # # 5.6
    # input_data = [[[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]],
    #               [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],]

    # mask = [[[1, -0.1], [1, -0.1]],
    #         [[0.5, 0.5], [-0.5, -0.5]],]

    # input_data_tensor = torch.tensor(np.expand_dims(input_data, axis=0))
    # mask_tensor = torch.tensor(np.expand_dims(mask, axis=0))

    # result = F.conv2d(input_data_tensor, mask_tensor, stride=1, padding=0).numpy()
    # print("The result of convolutional is:\n", result)

    # result = F.conv2d(input_data_tensor, mask_tensor, stride=1, padding=1).numpy()
    # print("The result of convolutional is:\n", result)

    # result = F.conv2d(input_data_tensor, mask_tensor, stride=2, padding=1).numpy()
    # print("The result of convolutional is:\n", result)

    # result = F.conv2d(input_data_tensor, mask_tensor, stride=1, padding=0, dilation=2).numpy()
    # print("The result of convolutional is:\n", result)

    # # 5.7 
    # input_data = [[[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]],
    #               [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],
    #               [[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]],]

    # mask = [[[1]],
    #         [[-1]],
    #         [[0.5]]]
    # input_data_tensor = torch.tensor(np.expand_dims(input_data, axis=0))
    # mask_tensor = torch.tensor(np.expand_dims(mask, axis=0))       
    # result = F.conv2d(input_data_tensor, mask_tensor, stride=1, padding=0).numpy()
    # print("The result of convolutional is:\n", result)

    # # 5.8
    # input_data = [[0.2, 1, 0, 0.4],
    #               [-1, 0, -0.1, -0.1],
    #               [0.1, 0, -1, -0.5],
    #               [0.4, -0.7, -0.5, 1]]
    # input_data_tensor = torch.tensor(np.expand_dims(input_data, axis=(0, 1)))

    # result = F.avg_pool2d(input_data_tensor, kernel_size=(2, 2), stride=2).numpy()
    # print("The result of avg pool is:\n", result)

    # result = F.max_pool2d(input_data_tensor, kernel_size=(2, 2), stride=2).numpy()
    # print("The result of max pool is:\n", result)

    # result = F.max_pool2d(input_data_tensor, kernel_size=(3, 3), stride=1).numpy()
    # print("The result of max pool is:\n", result)

    # # 5.9

    # # height, width, channel, sample
    # input_dim = [11, 15, 6, 1]
    # # height, width, channel, numbers
    # mask_dim = [3, 3, 6, 1]
    
    # padding = 0
    # stride = 2

    # result = compute_conv_output_size(input_dim, mask_dim, stride=stride, padding=padding)
    # print("the output size of convolutional layer is:\n", result)

    # # 5.10

    # shape_1 = compute_conv_output_size(input_size=[200, 200, 3, 1], mask_size=[5, 5, 3, 40], stride=1, padding=0)
    # print("the output size of convolutional layer is:\n", shape_1)

    # shape_2 = compute_pool_output_size(shape_1, mask_size=(2, 2), stride=2)
    # print("the output size of pool layer is:\n", shape_2)

    # shape_3 = compute_conv_output_size(shape_2, mask_size=(4, 4, shape_2[2], 80), stride=2, padding=1)
    # print("the output size of convolutional layer is:\n", shape_3)

    # shape_4 = compute_conv_output_size(shape_3, mask_size=(1, 1, shape_3[2], 20), stride=1, padding=0)
    # print("the output size of convolutional layer is:\n", shape_4)

    # 7.4 7.6

    # # each row is a data point
    # data = np.array([[1, 2, 1],
    #         [2, 3, 1],
    #         [3, 5, 1],
    #         [2, 2, 1]])

    # # data = np.array([[0, 1],
    # #                  [3, 5],
    # #                  [5, 4],
    # #                  [5, 6],
    # #                  [8, 7],
    # #                  [9, 7]])

    # new_data = KL_transform(data.T, dimension=2)

    # # each row is a new data point
    # print("new data is:\n", new_data.T)

    # 7.7
    
    # # each row is a data point
    # x = np.array([[0.0, 1.0],
    #                  [3.0, 5.0],
    #                  [5.0, 4.0],
    #                  [5.0, 6.0],
    #                  [8.0, 7.0],
    #                  [9.0, 7.0]])
    # w =  [-1.0, 0.0]
    # lr = 0.01

    # model = Batch_oja_learning(x=x, w=w, lr=lr)
    # model.train(epoch=6, log=True)

    # # 7.10
    # data_array = np.array([[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]])
    # class_array = np.array([1, 1, 1, 2, 2])
    # w_array = np.array([[-1, 5], [2, -3]])

    # Fisher_method(data_array=data_array, class_array=class_array, w_array=w_array)

    
    # # 7.11
    # V = np.array([[-0.62, 0.44, -0.91],
    #      [-0.81, -0.09, 0.02],
    #      [0.74, -0.91, -0.60],
    #      [-0.82, -0.92, 0.71],
    #      [-0.26, 0.68, 0.15],
    #      [0.80, -0.94, -0.83]])
    # X = np.array([[1, 0, 0],
    #      [1, 0, 1],
    #      [1, 1, 0],
    #      [1, 1, 1]])
    # W = np.array([0, 0, 0, -1, 0, 0, 2])
    # Y = Heaviside(np.dot(V, X.T), thres=0)
    # print("The result of Y is:\n", Y)
    # Y = np.insert(Y, 0, np.array([1, 1, 1, 1]), axis=0)
    # Z = np.dot(W, Y)
    # print("The result of Z is:\n", Z)

    # # 7.12
    # sparse_code_array = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
    #                               [0, 0, 1, 0, 0, 0, -1, 0],
    #                               [0, 0, 0, -1, 0, 0, 0, 0]])
    # dictionary = np.array([[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
    #                        [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]])
    # x = np.array([-0.05, -0.95])

    # select_sparser_code(sparse_code_array=sparse_code_array, dictionary=dictionary, x=x)

    # # 10.1
    # feature_vector_array = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    # k = 2
    # ori_feature_vetor_array = [[-1, 3], [5, 1]]
    # k_means(feature_vector_array=feature_vector_array, k=2, ori_feature_vetor_array=ori_feature_vetor_array, method='eucli')

    # # 10.3
    # data = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    # centers = [[-0.5, 1.5], [0, 2.5], [1.5, 0]]
    # k = 3
    # lr = 0.1
    # sample_order = [3, 1, 1, 5, 6]

    # model = Competitive_learning(data=data, centers=centers, k=k, lr=lr, sample_order=sample_order)
    # model.train(log=True)

    # # 10.4
    # data = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    # lr = 0.5
    # thres = 3
    # sample_order = [3, 1, 1, 5, 6]

    # model = Basic_leader_follower(data=data, lr=lr, thres=thres, sample_order=sample_order)
    # model.train(log=True)

    # # 10.5
    # data = np.array([[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]).T
    # init = np.array([[1, 0.5, 0.5, 0.5, 0.5, 0], [0, 0.5, 0.5, 0.5, 0.5, 1]])
    # result = cmeans(data=data, c=2, m=2, error=0.2, maxiter=4, init=init)
    # print(result[0])

    # # 10.6
    # data = [[-1, 3], [1, 2], [0, 1], [4, 0], [5, 4], [3, 2]]

    # # xme
    # agglomerative_hierarchical_clusteringv2(data, k=3, method='eucli', cluster_method='single_link')

    # # sklearn
    # clustering = AgglomerativeClustering(n_clusters=3, linkage='single', compute_distances=True).fit(data)
    # print(clustering.labels_)

    #