# -*- encoding: utf-8 -*-
'''
@File    :   radial_basis_function.py
@Time    :   2021/05/23 19:39:56
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


import numpy as np
from base import ModelBase
from collections import defaultdict
from tabulate import tabulate
from metrics import compute_l2_diff


class Radial_basis_function(ModelBase):
    def __init__(self, input_data, target_data, centers, stds):
        self.input_data = input_data
        self.target_data = target_data
        self.centers = centers
        self.stds = stds
        self.build_net()

    def build_net(self):
        hidden_output_list = []
        for i in range(len(self.centers)):
            hidden_output = np.exp(-np.sum(np.square(self.input_data - self.centers[i]), axis=1)/ (2 * np.square(self.stds[i])))
            hidden_output_list.append(hidden_output)
        print("the output of hidden layer is \n{}".format(np.array(hidden_output_list).T))
        hidden_output_list.append([ 1 for _ in range(len(self.input_data))])
        hidden_output_array = np.array(hidden_output_list)
        
        self.w_array = np.dot(np.linalg.pinv(hidden_output_array).T, self.target_data.T)
        print("the result of the weight array is \n{}".format(self.w_array))

    def pred(self, test_data, gt_value=None):
        hidden_output_list = []
        for i in range(len(self.centers)):
            hidden_output = np.exp(-np.sum(np.square(test_data - self.centers[i]), axis=1)/ (2 * np.square(self.stds[i])))
            hidden_output_list.append(hidden_output)
        print("the hidden output of test data is \n{}".format(np.array(hidden_output_list).T))
        hidden_output_list.append([ 1 for _ in range(len(self.input_data))])
        hidden_output_array = np.array(hidden_output_list)
        
        pred_value = np.dot(hidden_output_array.T, self.w_array)
        print("the pred value of test data is \n{}".format(pred_value))

        if gt_value.any():
            print("the error is \n{}".format(self.compute_error(pred_value, gt_value)))
    
    def compute_error(self, pred_value, gt_value):
        return compute_l2_diff(pred_value, gt_value.T)


        

if __name__ == "__main__":
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

    input_data = np.array([[0.05], [0.2], [0.25], [0.3], [0.4], [0.43], [0.48], [0.6], [0.7], [0.8], [0.9], [0.95]])
    target_data = np.array([[0.0863, 0.2662, 0.2362, 0.1687, 0.1260, 0.1756, 0.3290, 0.6694, 0.4573, 0.3320, 0.4063, 0.3535]])
    # centers = np.array([[0.2], [0.6], [0.9]])
    # stds = np.array([0.1, 0.1, 0.1])
    # net = Radial_basis_function(input_data, target_data, centers, stds)

    centers = np.array([[0.1667], [0.35], [0.5525], [0.8833]])
    stds = np.array([0.7842, 0.7842, 0.7842, 0.7842])
    net = Radial_basis_function(input_data, target_data, centers, stds)
    net.pred(test_data=input_data, gt_value=target_data)