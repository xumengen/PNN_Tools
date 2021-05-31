# -*- encoding: utf-8 -*-
'''
@File    :   agglomerative_clustering.py
@Time    :   2021/05/13 15:44:48
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from metrics import compute_SAD_diff, compute_eucli_diff


def compute_dist_between_two_feature_vector_array(array_1, array_2, method='SAD'):
        """
        Args:
        array_1: m1*n
        array_2: m2*n
        Return: m1*m2
        """

        array_1 = np.array(array_1)
        array_2 = np.array(array_2)
        assert len(array_1.shape) == 2 and len(array_2.shape) == 2
        result_array = list()
        for i in range(array_1.shape[0]):
            dist = list()
            for j in range(array_2.shape[0]):
                if method == 'SAD':
                    dist.append(compute_SAD_diff(array_1[i], array_2[j]))
                elif method == 'eucli':
                    dist.append(compute_eucli_diff(array_1[i], array_2[j]))
            result_array.append(dist[:])
        return np.array(result_array)

def agglomerative_hierarchical_clusteringv2(feature_vector_array, k=3, method='SAD', cluster_method='centroid'):
        """
        """

        feature_vector_array = np.array(feature_vector_array)
        cluster_array = feature_vector_array.reshape(-1, 1, feature_vector_array.shape[-1])
        cluster_list = cluster_array.tolist()
        old_cluster_array = cluster_array.copy()

        record_list = []
        for i in range(1, cluster_array.shape[0]+1):
            record_list.append([i])
        step = 1
        while len(record_list) > k:
            result_list = []
            for i in range(len(record_list)):
                cluster_1 = np.array(cluster_list[i])
                for j in range(i+1, len(record_list)):
                    cluster_2 = np.array(cluster_list[j])
                    if cluster_method == 'centroid':
                        dist = compute_SAD_diff(cluster_1, cluster_2)
                    elif cluster_method == 'single_link':
                        dist_array = compute_dist_between_two_feature_vector_array(cluster_1, cluster_2, method)
                        dist = np.min(dist_array)
                    elif cluster_method == 'complete_link':
                        dist_array = compute_dist_between_two_feature_vector_array(cluster_1, cluster_2, method)
                        dist = np.max(dist_array)
                    elif cluster_method == 'group_average':
                        dist_array = compute_dist_between_two_feature_vector_array(cluster_1, cluster_2, method)
                        dist = np.mean(dist_array)
                    result_list.append([i+1, j+1, dist])
            min_dist = min([i[-1] for i in result_list])
            cluster_result = []
            pair = [0, 0]
            for result in result_list:
                if result[-1] == min_dist:
                    pair[0] = result[0]
                    pair[1] = result[1]
                    symbol = False
                    for idx, cluster_info in enumerate(cluster_result):
                        if pair[0] in cluster_info or pair[1] in cluster_info:
                            cluster_info.extend([pair[0], pair[1]])
                            cluster_result[idx] = list(set(cluster_info))
                            symbol = True
                            break
                    if not symbol:
                        cluster_result.append([pair[0], pair[1]])
            new_cluster_index = list()
            for result in cluster_result:
                for i in result:
                    new_cluster_index.append(i)
            new_cluster_list = list()
            new_record_list = list()
            for i in range(len(cluster_result)):
                if not cluster_result[i]:
                    continue
                tmp = []
                for result in cluster_result[i]:
                    target = record_list[result-1]
                    tmp.extend(target)
                new_record_list.append(tmp)
            for i in range(1, len(record_list)+1):
                if i in new_cluster_index:
                    continue
                else:
                    new_record_list.append(record_list[i-1])
            for i in range(len(new_record_list)):
                cluster_index_list = np.array(new_record_list[i]) - 1
                if cluster_method == 'centroid':
                    new_cluster_list.append(np.mean(old_cluster_array[cluster_index_list], axis=0))
                else:
                    tmp_cluster_list = []
                    for j in cluster_index_list:
                        tmp_cluster_list.append(np.squeeze(old_cluster_array[j]))
                    new_cluster_list.append(tmp_cluster_list)
            cluster_list = new_cluster_list
            record_list = new_record_list
            print("the feature of cluster of {}-th step is\n {}\n".format(step, np.array(cluster_list)))
            print("the result of cluster of {}-th step is\n {}\n".format(step, record_list))
            step += 1

        print("the result of agglomerative hierarchical clustering is\n {}\n".format(record_list))
        return record_list


if __name__ == "__main__":
    data = [[-1, 3], [1, 2], [0, 1], [4, 0], [5, 4], [3, 2]]

    # xme
    agglomerative_hierarchical_clusteringv2(data, k=3, method='eucli', cluster_method='single_link')

    # sklearn
    clustering = AgglomerativeClustering(n_clusters=3, linkage='single', compute_distances=True).fit(data)
    print(clustering.labels_)