# -*- encoding: utf-8 -*-
'''
@File    :   k_means.py
@Time    :   2021/05/12 17:37:26
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np
from metrics import compute_SAD_diff, compute_eucli_diff
from tabulate import tabulate


def k_means(feature_vector_array, k, ori_feature_vetor_array, method='SAD'):
        """
    
        """
        def compute_new_cluster_center():
            new_ori_feature_vetor_array = np.zeros(ori_feature_vetor_array.shape)
            for i in range(1, k+1):
                correspond_index = np.where(result_array==i)
                new_ori_feature_vetor_array[i-1][:] = np.mean(feature_vector_array[correspond_index], axis=0)
            return new_ori_feature_vetor_array

        assert k > 0
        assert k == len(ori_feature_vetor_array)
        feature_vector_array = np.array(feature_vector_array)
        ori_feature_vetor_array = np.array(ori_feature_vetor_array)
        result_array = np.zeros(feature_vector_array.shape[:-1])

        count = 1
        while True:
            result_array_copy = result_array.copy()
            all_dist_list = []
            for i in range(feature_vector_array.shape[0]):
                feature_vector = feature_vector_array[i]
                assert feature_vector.shape == ori_feature_vetor_array[0].shape
                dist_list = list()
                for m in range(k):
                    if method == 'SAD':
                        dist_list.append(compute_SAD_diff(feature_vector, ori_feature_vetor_array[m]))
                    elif method == 'eucli':
                        dist_list.append(compute_eucli_diff(feature_vector, ori_feature_vetor_array[m]))
                label = dist_list.index(min(dist_list)) + 1
                result_array[i] = label
                all_dist_list.append(dist_list)
            print("the result of {}-th step is\n {}".format(count, result_array))
            print("the result of distance between points and centroids is {}".format(all_dist_list))
            all_dist_list = []
            if (result_array_copy == result_array).all():
                break
            count += 1
            ori_feature_vetor_array = compute_new_cluster_center()
            print("the cluster of {}-th step is\n {}".format(count, ori_feature_vetor_array))

        print("the result of region k-means is\n {}\n".format(result_array))
        return result_array


if __name__ == "__main__":
    feature_vector_array = [[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]]
    k = 2
    ori_feature_vetor_array = [[-1, 3], [5, 1]]
    k_means(feature_vector_array=feature_vector_array, k=2, ori_feature_vetor_array=ori_feature_vetor_array, method='eucli')