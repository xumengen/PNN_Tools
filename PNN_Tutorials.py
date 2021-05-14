import numpy as np
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


# Tutorial 2.1 dichotomy
# def linear_discriminant(w, x, w0):
#    g = np.dot(w, x) + w0
#    return g


# W = np.array([2, 1])
# W0 = -5
# X = np.array([1, 1])
# g_x = linear_discriminant(W, X, W0)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.2, 2.5 dichotomy in augmented feature space
# def aug_linear_discriminant(a, x):
#    y = np.insert(x, 0, 1)
#    g = np.dot(a, y)
#    return g


# a_t = np.array([-3, 1, 2, 2, 2, 4])
# X = np.array([1, 1, 1, 1, 1])
# g_x = aug_linear_discriminant(a_t, X.T)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.3 3D quadratic discriminant
# def quadratic_discriminant_3d(x):
#    g = pow(x[0], 2) - pow(x[2], 2) + 2 * x[1] * x[2] + 4 * x[0] * x[1] + 3 * x[0] - 2 * x[1] + 2
#    return g


# X = np.array([1, 1, 1])
# g_x = quadratic_discriminant_3d(X)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.4 2D quadratic discriminant
# def quadratic_discriminant_2d(a, b, c, x):
#    g = np.dot(np.dot(x, a), x.T) + np.dot(x, b.T) + c
#    return g


# A = np.array([[-2, 5], [5, -8]])
# B = np.array([1, 2])
# C = -3
# X = np.array([1, 1])
# g_x = quadratic_discriminant_2d(A, B, C, X)
# if g_x > 0:
#    print("The class of this vector is {}.".format(1))
# else:
#    print("The class of this vector is {}.".format(2))


# Tutorial 2.6 batch perceptron y的第一位永远是1，注意题目有没有说sample normalisation，如果有，则第二类实例对应的y乘以-1,并且a的更新条件发生变化
# def batch_perceptron_with_sample_normalisation(a, lr, y):
#    while True:
#        miss_class = []
#        y_mis = []
#        temp = 0
#        for i in range(len(y)):
#            g = np.dot(a, y[i].T)
#            if g > 0:
#                miss_class.append('no')
#            else:
#                miss_class.append('yes')
#                y_mis.append(y[i])
#        if miss_class.count('no') != len(miss_class):
#            for j in range(len(y_mis)):
#                temp += lr * y_mis[j]
#            a += np.array(temp)
#        else:
#            break
#    return a


# def batch_perceptron(a, lr, y, labels):
#    while True:
#        pred_label = []
#        y_mis = []
#        temp = 0
#        for i in range(len(y)):
#            g = np.dot(a, y[i].T)
#            if g > 0:
#                pre_label = 1
#            else:
#                pre_label = 2
#            pred_label.append(pre_label)
#            if pre_label != labels[i]:
#                y_mis.append(y[i])
#        if pred_label == labels:
#            break
#        else:
#            for j in range(len(y_mis)):
#                temp += lr * y_mis[j]
#            a += np.array(temp)
#    return a

# a_t = np.array([-25, 6, 3])
# rate = 1
# yk = np.array([[1, 1, 5],
#               [1, 2, 5],
#               [-1, -4, -1],
#               [-1, -5, -1]])
# label = [1, 1, 2, 2]
# a_final = batch_perceptron_with_sample_normalisation(a_t, rate, yk)
# a_final = batch_perceptron(a_t, rate, yk, label)
# print("The value of a after learning is: {}.".format(a_final))


# Tutorial 2.7, 2.9, 2.10 sequential perceptron y的第一位永远是1，注意题目有没有说sample normalisation，如果有，则第二类实例对应的y乘以-1，并且a的更新条件和更新等式都发生变化
# def sequential_perceptron(a, lr, yk, labels):
#    while True:
#        pred_label = []
#        for i in range(len(yk)):
#            g = np.dot(a, yk[i].T)
#            if g > 0:
#                pre_label = 1
#            else:
#                pre_label = -1
#            pred_label.append(pre_label)
#            if pre_label != labels[i]:
#                a = a + lr * labels[i] * yk[i]
#        if pred_label == labels:
#            break
#    return a


# def sequential_perceptron_with_sample_normalisation(a, lr, yk):
#    while True:
#        count = 0
#        for i in range(len(yk)):
#            g = np.dot(a, yk[i].T)
#            if g <= 0:
#                a = a + lr * yk[i]
#            else:
#                count += 1
#        if len(yk) == count:
#            break
#    return a


# if __name__ == '__main__':
#    a_t = np.array([-25, 6, 3])
#    rate = 1
#    data_set = np.array([[1, 1, 5],
#                         [1, 2, 5],
#                         [-1, -4, -1],
#                         [-1, -5, -1]])
#    label = [1, 1, 1, -1, -1, -1]
#    a_final = sequential_perceptron(a_t, rate, data_set, label)
#    a_final = sequential_perceptron_with_sample_normalisation(a_t, rate, data_set)
#    print("The value of a after learning is: {}.".format(a_final))


# Tutorial 2.11 sequential multiclass perceptron
# def sequential_multiclass_perceptron(a, y, lr, labels):
#    n = 0
#    while True:
#        for i in range(len(y)):
#            g1 = np.dot(a[0], y[i])
#            g2 = np.dot(a[1], y[i])
#            g3 = np.dot(a[2], y[i])
#            if g1 == g2 == g3:
#                a[2] -= lr * y[i]
#                a[labels[i]-1] += lr * y[i]
#            if g1 > g2 > g3 or g1 > g3 > g2 or g1 > g2 == g3:
#                if labels[i] != 1:
#                    a[labels[i]-1] += lr * y[i]
#                    a[0] -= lr * y[i]
#            elif g2 > g1 > g3 or g2 > g3 > g1 or g2 > g1 == g3:
#                if labels[i] != 2:
#                    a[labels[i]-1] += lr * y[i]
#                    a[1] -= lr * y[i]
#            elif g3 > g1 > g2 or g3 > g2 > g1 or g3 > g1 == g2:
#                if labels[i] != 3:
#                    a[labels[i]-1] += lr * y[i]
#                    a[2] -= lr * y[i]
#            n += 1
#            print("n = {}, a1 = {}, a2 = {}, a3 = {}".format(n, a[0], a[1], a[2]))


# a_t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# y_t = np.array([[1, 1, 1], [1, 2, 0], [1, 0, 2], [1, -1, 1], [1, -1, -1]])
# rate = 1
# label = [1, 1, 2, 2, 3]
# sequential_multiclass_perceptron(a_t, y_t, rate, label)


# Tutorial 2.12, 2.13 pseudoinverse with sample normalisation
# Y = np.array([[1, 0, 2],
#              [1, 1, 2],
#              [1, 2, 1],
#             [-1, 3, -1],
#             [-1, 2, 1],
#              [-1, 3, 2]])
# b = np.array([1, 1, 1, 2, 2, 2])
# a = np.dot(np.linalg.pinv(Y), b.T)
# print("a = {}".format(a))


# Tutorial 2.15 KNN
# knn = KNeighborsClassifier(n_neighbors=3)  # Define KNN classier model (n_neighbors represents the value of k)
# y = np.array([1, 2, 2, 3, 3])  # specify prediction target
# X = np.array([[0.15, 0.35],
#              [0.15, 0.28],
#              [0.12, 0.2],
#              [0.1, 0.32],
#              [0.06, 0.25]])  # choose features, y = f(X)
# knn.fit(X, y)  # fit the model (the heart of modelling)
# print(knn.predict([[0.1, 0.25]]))  # using this model to predict


# Tutorial 3.2
# def heaviside(w, x, threshold):
#    y = np.dot(w, x.T) - threshold
#    if y > 0:
#        print("The output of neuron is: {}".format(1))
#    else:
#        print("The output of neuron is: {}".format(0))


# W = np.array([0.1, -0.5, 0.4])
# thresh = 0
# x1 = np.array([0.1, -0.5, 0.4])
# x2 = np.array([0.1, 0.5, 0.4])
# heaviside(W, x2, thresh)


# Tutorial 3.3, 3.5, 3.6
# w = [w0, w1, ..., wn], w0=-theta
# def sequential_delta(w, lr, t, x):
#    n = 0
#    while True:
#        y = []
#        for i in range(len(x)):
#            h = np.dot(w, np.insert(x[i].T, 0, 1))
#            if h > 0:
#                y.append(1)
#            else:
#                y.append(0)
#            w = w + lr * (t[i] - y[i]) * np.insert(x[i], 0, 1)
#            n += 1
#            print("n = {}, y = {}, w = {}".format(n, y[i], w))
#        if y == t:
#            break


# Tutorial 3.4
# def batch_delta(w, lr, t, x):
#    epoch = 0
#    while True:
#        y = []
#        Temp = []
#        for i in range(len(x)):
#            h = np.dot(w, np.insert(x[i].T, 0, 1))
#            if h > 0:
#                y.append(1)
#            else:
#                y.append(0)
#            temp = lr * (t[i] - y[i]) * np.insert(x[i], 0, 1)
#            Temp.append(temp)
#        w += sum(Temp)
#        epoch += 1
#        print("epoch = {}, w = {}".format(epoch, w))
#        if y == t:
#            break


# thresh = -1
# w1 = 0
# w2 = 0
# rate = 1
# W = np.array([-thresh, w1, w2])
# T = [1, 1, 1, 0, 0, 0]
# X = np.array([[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]])
# sequential_delta(W, rate, T, X)


# Tutorial 3.7, 3.8
# def negative_feedback(x, w, alpha, n):
#    y = np.array([0, 0])
#    for i in range(0, n):
#        e = x.T - np.dot(w.T, y.T)
#        y = y + alpha * np.dot(w, e)
#        print("i = {}, the activation of the output neurons: {}".format(i+1, y))


# Tutorial 3.9
# def negative_feedback_stable(x, w1, w2, n, e1, e2):
#    y = np.array([0, 0])
#    for i in range(0, n):
#        a = []
#        b = []
#        temp = np.dot(w1.T, y.T)
#        for j in range(len(temp)):
#            a.append(max(temp[j], e2))
#        e = x.T / np.array(a)
#        for k in range(len(y)):
#            b.append(max(y[k], e1))
#        y = np.array(b) * np.dot(w2, e)
#        print("i = {}, the activation of the output neurons: {}".format(i + 1, y))


# X = np.array([1, 1, 0])
# Alpha = 0.5
# W = np.array([[1, 1, 0],
#              [1, 1, 1]])
# w = np.array([[1/2, 1/2, 0],
#              [1/3, 1/3, 1/3]])
# count = 5
# e_1 = 0.01
# e_2 = 0.01
# negative_feedback_stable(X, W, w, count, e_1, e_2)


# Tutorial 4.1 total connection weights
# def total_connection_weights(num_in, num_out, hidden, num_hid, bias):
#    num_weights = 0
#    if bias:
#        for i in range(num_hid):
#            if i == 0:
#                num_weights += (num_in + 1) * hidden[i]
#            else:
#                num_weights += ((hidden[i - 1]) + 1) * hidden[i]
#        num_weights += (hidden[num_hid - 1] + 1) * num_out
#    else:
#        for i in range(num_hid):
#            if i == 0:
#                num_weights += num_in * hidden[i]
#            else:
#                num_weights += hidden[i - 1] * hidden[i]
#        num_weights += hidden[num_hid - 1] * num_out
#    return num_weights


# n_in = 2
# n_out = 3
# hid = [4, 5]
# num_hidden = 2
# Bias = True
# print("Total number of weights:", total_connection_weights(n_in, n_out, hid, num_hidden, Bias))


# Tutorial 4.4 represent patterns
# def represent_patterns(w1, w2, w3, w4, x):
#    temp_y = np.dot(w1, x) + w2
#    y = 2 / (1 + 1 / pow(math.e, 2 * temp_y)) - 1
#    temp_z = np.dot(w3, y) + w4
#    z = 1 / (1 + 1 / (pow(math.e, temp_z)))
#    print("y = {}, z = {}".format(y, z))
#    print("This pattern is represented by", [round(z[0]), round(z[1])])


# W1 = np.array([[-0.7057, 1.9061, 2.6605, -1.1359],
#               [0.4900, 1.9324, -0.4269, -5.1570],
#               [0.9438, -5.4160, -0.3431, -0.2931]])
# W2 = np.array([4.8432, 0.3973, 2.1761])
# W3 = np.array([[-1.1444, 0.3115, -9.9812], [0.0106, 11.5477, 2.6479]])
# W4 = np.array([2.5230, 2.6463])
# X = np.array([1, 1, 0, 0])
# represent_patterns(W1, W2, W3, W4, X)


# Tutorial 4.5
# Q.a
# w = np.array([[0.5, 0], [0.3, -0.7]])
# w0 = np.array([0.2, 0])
# x = np.array([0.1, 0.9])
# temp_y = np.dot(w, x.T) + w0
# y = 2 / (1 + 1 / pow(math.e, 2 * temp_y)) - 1
# temp_z = 0.8 * y[0] + 1.6 * y[1] - 0.4
# z = 2 / (1 + 1 / pow(math.e, 2 * temp_z)) - 1
# print("y = {}, z = {}".format(y, z))

# Q.c
# lr = 0.25
# t = 0.5
# m10 = -0.4
# temp_m = -lr * (z - t) * (4 / pow(math.e, 2 * temp_z)) / pow(1 + 1 / pow(math.e, 2 * temp_z), 2)
# m10 += temp_m
# w21 = 0.3
# w22 = -0.7
# m12 = 1.6
# a = w21 * x[0] + w22 * x[1]
# temp_w = -lr * (z - t) * ((4 / pow(math.e, 2 * temp_z)) / pow(1 + 1 / pow(math.e, 2 * temp_z), 2)) * m12 * ((4 / pow(math.e, 2 * a)) / pow(1 + 1 / pow(math.e, 2 * a), 2)) * x[0]
# w21 += temp_w
# print("m10 = {}, w21 = {}".format(m10, w21))


# Tutorial 4.6
# Q.b
# c1 = np.array([0, 0])
# c2 = np.array([1, 1])
# c = abs(c1 - c2)
# n_H = 2
# x = np.array([1.8, 0.6])
# sd = np.linalg.norm(c) / math.sqrt(2 * n_H)
# d1 = abs(x - c1)
# y1 = pow(math.e, -pow(np.linalg.norm(d1), 2) / (2 * pow(sd, 2)))
# d2 = abs(x - c2)
# y2 = pow(math.e, -pow(np.linalg.norm(d2), 2) / (2 * pow(sd, 2)))
# print("y1 = {}, y2 = {}".format(y1, y2))

# Q.c least squares method
# 代入不同的x，分别计算出y1和y2；注意fai_x是square matrix和non-square matrix的计算公式不同，别忘了最后一列的1
# fai_x = np.array([[0.7711, 0.2322, 1], [0.2276, 0.2276, 1], [0.4819, 0.5886, 1], [0.0273, 0.4493, 1]])
# t = np.array([0, 1, 1, 0])
# temp1 = np.dot(fai_x.T, t)
# temp2 = np.dot(fai_x.T, fai_x)
# W = np.dot(np.linalg.inv(temp2), temp1)
# print("The output weights using the least squares method is:", W)

# Q.d determine classes
# W = [-2.50312891, -2.50312891, 2.84180225]
# z = []
# Class = []
# for i in range(len(fai_x)):
#     z.append(W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2])
# for i in range(len(z)):
#     if z[i] > 0.5:
#         Class.append(1)
#     else:
#         Class.append(0)
# for i in range(len(Class)):
#    print(Class[i], end=' ')


# Tutorial 4.7
# Q.a
# The answer of it is located at Jupyter Notebook

# Q.c, Q.d, Q.e
# Q.d 受小数位精度影响，输出结果的小数位和答案的小数位不同
# Q.e 输出结果与答案不同，应该是答案的问题
# c1, c2, c3 = 0.2, 0.6, 0.9
# sd = 0.1
# x = [0.05, 0.2, 0.25, 0.3, 0.4, 0.43, 0.48, 0.6, 0.7, 0.8, 0.9, 0.95]
# t = np.array([0.0863, 0.2662, 0.2362, 0.1687, 0.1260, 0.1756, 0.3290, 0.6694, 0.4573, 0.3320, 0.4063, 0.3535])
# c1 = (x[0] + x[1] + x[2]) / 3
# c2 = (x[3] + x[4]) / 2
# c3 = (x[5] + x[6] + x[7] + x[8]) / 4
# c4 = (x[9] + x[10] + x[11]) / 3
# avg = (abs(c1-c2) + abs(c1-c3) + abs(c1-c4) + abs(c2-c3) + abs(c2-c4) + abs(c3-c4)) / 6
# sd = 2 * avg
# fai_x = []
# for i in range(len(x)):
#    y1 = pow(math.e, -pow(x[i] - c1, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y1)
#    y2 = pow(math.e, -pow(x[i] - c2, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y2)
#    y3 = pow(math.e, -pow(x[i] - c3, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y3)
#    y4 = pow(math.e, -pow(x[i] - c4, 2) / (2 * pow(sd, 2)))
#    fai_x.append(y4)
#    fai_x.append(1)
#    print("p = {}, y1 = {}, y2 = {}, y3 = {}".format(i + 1, y1, y2, y3))
#    print("p = {}, y1 = {}, y2 = {}, y3 = {}, y4 = {}".format(i + 1, y1, y2, y3, y4))
# fai_x = np.array(fai_x)
# fai_x = fai_x.reshape((12, 4))
# temp1 = np.dot(fai_x.T, t)
# temp2 = np.dot(fai_x.T, fai_x)
# W = np.dot(np.linalg.inv(temp2), temp1)
# print("The output weights are:", W)
# J = []
# for i in range(len(x)):
#    z = W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2] * fai_x[i][2] + W[3] * fai_x[i][3] + W[4]
#    z = W[0] * fai_x[i][0] + W[1] * fai_x[i][1] + W[2] * fai_x[i][2] + W[3]
#    J.append(pow((z - t[i]), 2))
#    J[i] = round(J[i], ndigits=4)
#    print(J[i])
# print("The sum of squared errors is:", sum(J))


# Tutorial 5.4 activation functions
# ReLU
# net = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# for i in range(len(net)):
#    for j in range(len(net[i])):
#        if net[i][j] < 0:
#            net[i][j] = 0
# print(net)

# LReLU when a = 0.1
# a = 0.1
# for i in range(len(net)):
#    for j in range(len(net[i])):
#        if net[i][j] < 0:
#            net[i][j] = a * net[i][j]
# print(net)

# tanh
# for i in range(len(net)):
#    for j in range(len(net[i])):
#            net[i][j] = np.tanh(net[i][j])
# print(net)

# Heaviside function with 0.1 threshold and define H(0) as 0.5
# for i in range(len(net)):
#    for j in range(len(net[i])):
#        if net[i][j] - 0.1 > 0:
#            net[i][j] = 1
#        elif net[i][j] - 0.1 < 0:
#            net[i][j] = 0
#        elif net[i][j] - 0.1 == 0:
#            net[i][j] = 0.5
# print(net)


# Tutorial 5.5 batch normalisation
# def batch_normalisation(x1, x2, x3, x4, beta, r, e):
#    bn1, bn2, bn3, bn4 = [], [], [], []
#    for i in range(len(x1)):
#        for j in range(len(x1[i])):
#            avg = np.mean([x1[i][j], x2[i][j], x3[i][j], x4[i][j]])
#            var = np.var([(x1[i][j] - avg), (x2[i][j] - avg), (x3[i][j] - avg), (x4[i][j] - avg)])
#            temp = beta + r * (x1[i][j] - avg) / math.sqrt(var + e)
#            bn1.append(temp)
#            temp = beta + r * (x2[i][j] - avg) / math.sqrt(var + e)
#            bn2.append(temp)
#            temp = beta + r * (x3[i][j] - avg) / math.sqrt(var + e)
#            bn3.append(temp)
#            temp = beta + r * (x4[i][j] - avg) / math.sqrt(var + e)
#            bn4.append(temp)
#    bn1 = np.array(bn1).reshape(3, 3)
#    bn2 = np.array(bn2).reshape(3, 3)
#    bn3 = np.array(bn3).reshape(3, 3)
#    bn4 = np.array(bn4).reshape(3, 3)
#    print("The Batch Normalisation of X1 is: \n {}".format(bn1))
#    print("The Batch Normalisation of X2 is: \n {}".format(bn2))
#    print("The Batch Normalisation of X3 is: \n {}".format(bn3))
#    print("The Batch Normalisation of X4 is: \n {}".format(bn4))


# b, r, e = 0, 1, 0.1
# x_1 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# x_2 = np.array([[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]])
# x_3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
# x_4 = np.array([[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]])
# batch_normalisation(x_1, x_2, x_3, x_4, b, r, e)


# Tutorial 5.6
# a. padding = 0, stride = 1
# X1 = np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]])
# X2 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# H1 = np.array([[1, -0.1], [1, -0.1]])
# H2 = np.array([[0.5, 0.5], [-0.5, -0.5]])
# out1 = []
# out2 = []
# for i in range(len(X1) - 1):
#    for j in range(len(X1)):
#        if j + 2 < len(X1) + 1:
#            out1.append(sum(sum(X1[i: i + 2, j: j + 2] * H1)))
#            out2.append(sum(sum(X2[i: i + 2, j: j + 2] * H2)))
#        else:
#            break
# out1 = np.array(out1).reshape(2, 2)
# out2 = np.array(out2).reshape(2, 2)
# out = out1 + out2
# print("The result of padding = 0 and stride = 1 is:\n", out)

# b. padding = 1, stride = 1
# X1_pad = np.pad(X1, 1)
# X2_pad = np.pad(X2, 1)
# out1_pad = []
# out2_pad = []
# for i in range(len(X1_pad) - 1):
#    for j in range(len(X1_pad)):
#        if j + 2 < len(X1_pad) + 1:
#            out1_pad.append(sum(sum(X1_pad[i: i + 2, j: j + 2] * H1)))
#            out2_pad.append(sum(sum(X2_pad[i: i + 2, j: j + 2] * H2)))
#        else:
#            break
# out1_pad = np.array(out1_pad).reshape(4, 4)
# out2_pad = np.array(out2_pad).reshape(4, 4)
# out_pad = out1_pad + out2_pad
# print("The result of padding = 1 and stride = 1 is:\n", out_pad)

# c. padding = 1, stride = 2 (2 step size along the width and height direction)
# using answer to b. part
# out_stride = [out_pad[0, 0], out_pad[0, 2], out_pad[2, 0], out_pad[2, 2]]
# out_stride = np.array(out_stride).reshape(2, 2)
# print("According to the answer to the b part, the result of padding = 1 and stride = 2 is:\n", out_stride)

# d. padding = 0, stride = 1, dilation = 2
# H1_d = np.array([[1, 0, -0.1], [0, 0, 0], [1, 0, -0.1]])
# H2_d = np.array([[0.5, 0, 0.5], [0, 0, 0], [-0.5, 0, -0.5]])
# out_d = X1 * H1_d + X2 * H2_d
# out_d = sum(sum(out_d))
# print("The result of padding = 0, stride = 1 and dilation = 2 is:\n", out_d)


# Tutorial 5.7
# x1 = np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]])
# x2 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
# x3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
# h = [1, -1, 0.5]
# y = []
# for i in range(len(x1)):
#    for j in range(len(x1)):
#        y.append(x1[i][j] * h[0] + x2[i][j] * h[1] + x3[i][j] * h[2])
# y = np.array(y).reshape(3, 3)
# print(y)


# Tutorial 5.8
# a. average pooling with a pooling region of 2-by-2 and stride = 2
# x = np.array([[0.2, 1, 0, 0.4], [-1, 0, -0.1, -0.1], [0.1, 0, -1, -0.5], [0.4, -0.7, -0.5, 1]])
# avg_pooling = []
# for i in range(0, len(x), 2):
#    for j in range(0, len(x), 2):
#        avg = np.mean(x[i: i + 2, j: j + 2])
#        avg_pooling.append(avg)
# avg_pooling = np.array(avg_pooling).reshape(2, 2)
# print("The result of average pooling is:\n", avg_pooling)

# b. max pooling with a pooling region of 2-by-2 and stride = 2
# max_pooling = []
# for i in range(0, len(x), 2):
#    for j in range(0, len(x), 2):
#        maximum = np.max(x[i: i + 2, j: j + 2])
#        max_pooling.append(maximum)
# max_pooling = np.array(max_pooling).reshape(2, 2)
# print("The result of max pooling with stride = 2 is:\n", max_pooling)

# c. max pooling with a pooling region of 3-by-3 and stride = 1
# max_pool = []
# for i in range(0, 2):
#    for j in range(0, 2):
#        maximum_ = np.max(x[i: i + 3, j: j + 3])
#        max_pool.append(maximum_)
# max_pool = np.array(max_pool).reshape(2, 2)
# print("The result of max pooling with stride = 2 is:\n", max_pool)


# Tutorial 5.9, 5.10
# number of masks = number of channels
# dim_feature_map = [98, 98]
# dim_mask = [4, 4]
# padding = 1
# stride = 2
# num_masks = 20
# outputHeight = 1 + (dim_feature_map[0] - dim_mask[0] + 2 * padding) / stride
# outputWidth = 1 + (dim_feature_map[1] - dim_mask[1] + 2 * padding) / stride
# print("The output size is:", (outputHeight, outputWidth, num_masks))
# len_feature_vec = outputHeight * outputWidth * num_masks
# print("after flattening, length of feature vector is:", len_feature_vec)


# Tutorial 6.3
# a. compute V(D, G)
# x1 = np.transpose(np.array([1, 2]))
# x2 = np.transpose(np.array([3, 4]))
# x_real = np.array([x1, x2])
# x_1 = np.transpose(np.array([5, 6]))
# x_2 = np.transpose(np.array([7, 8]))
# x_fake = np.array([x_1, x_2])
# theta_d1, theta_d2 = 0.1, 0.2
# E1 = 0.5 * math.log(pow(1 + pow(math.e, -(theta_d1 * x1[0] - theta_d2 * x1[1] - 2)), -1), math.e) + \
#    0.5 * math.log(pow(1 + pow(math.e, -(theta_d1 * x2[0] - theta_d2 * x2[1] - 2)), -1), math.e)
# E2 = 0.5 * math.log(1 - pow(1 + pow(math.e, -(theta_d1 * x_1[0] - theta_d2 * x_1[1] - 2)), -1), math.e) + \
#    0.5 * math.log(1 - pow(1 + pow(math.e, -(theta_d1 * x_2[0] - theta_d2 * x_2[1] - 2)), -1), math.e)
# V_D_G = E1 + E2
# print("The value of V(D, G) is:", V_D_G)


# Tutorial 7.4, 7.6 Karhunen-Loeve Transform
# x = np.array([[1, 2, 1],
#              [2, 3, 1],
#              [3, 5, 1],
#              [2, 2, 1]])
# x = np.array([[0, 1],
#              [3, 5],
#              [5, 4],
#              [5, 6],
#              [8, 7],
#              [9, 7]])
# num_samples, num_features = x.shape
# avg = np.array([np.mean(x[:, i]) for i in range(num_features)])
# x_norm = x - avg
# cov_matrix = np.dot(np.transpose(x_norm), x_norm)
# eig_val, eig_vec = np.linalg.eig(cov_matrix)
# print(eig_val)
# eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(num_features)]
# eig_pairs.sort(reverse=True)
# select the top k eigenvectors
# k = 1
# feature = np.array([ele[1] for ele in eig_pairs[:k]])
# get new data
# data = np.dot(x_norm, np.transpose(feature))
# print(data)  # the answer is still correct either or both columns can be multiplied by -1

# Tutorial 7.7 Oja's learning rule using the same data used in 7.6
# lr = 0.01
# w = np.array([-1.0, 0.0])
# epoch = 2
# for i in range(epoch):
#    temp = []
#    for j in range(len(x_norm)):
#        y = np.dot(w, x_norm[j].T)
#        temp.append(lr * y * (x_norm[j] - y * w))
#    total = 0
#    for k in range(len(temp)):
#        total += temp[k]
#    w += np.array(total)
#    print(w)

# Tutorial 7.10 Linear Discriminant Analysis Fisher's method
# w1 = np.array([-1, 5])
# w2 = np.array([2, -3])
# x = np.array([[1, 2],
#              [2, 1],
#              [3, 3],
#              [6, 5],
#              [7, 8]])
# x1 = x[:3]
# x2 = x[3:]
# num_class1, num_feature_class1 = x1.shape
# mean_class1 = np.array([np.mean(x1[:, i]) for i in range(num_feature_class1)])
# num_class2, num_feature_class2 = x2.shape
# mean_class2 = np.array([np.mean(x2[:, i]) for i in range(num_feature_class2)])
# sb1 = np.abs(np.dot(w1, np.transpose(mean_class1 - mean_class2))) ** 2
# sb2 = np.abs(np.dot(w2, np.transpose(mean_class1 - mean_class2))) ** 2
# count1_1, count1_2, count2_1, count2_2 = 0.0, 0.0, 0.0, 0.0
# for i in range(len(x1)):
#    count1_1 += np.dot(w1, np.transpose(x1[i] - mean_class1)) ** 2
#    count2_1 += np.dot(w2, np.transpose(x1[i] - mean_class1)) ** 2
# for i in range(len(x2)):
#    count1_2 += np.dot(w1, np.transpose(x2[i] - mean_class2)) ** 2
#    count2_2 += np.dot(w2, np.transpose(x2[i] - mean_class2)) ** 2
# sw1 = count1_1 + count1_2
# sw2 = count2_1 + count2_2
# J_w1 = sb1 / sw1
# J_w2 = sb2 / sw2
# print("The values of cost function J(w) for w1 and w2 are:", J_w1, J_w2)
# if J_w1 > J_w2:
#    print("w1 is a more effective projection weight.")
# elif J_w1 < J_w2:
#    print("w2 is a more effective projection weight.")
# else:
#    print("w1 and w2 are equally effective weights.")

# Tutorial 7.11 Extreme Learning Machine
# w = np.array([0, 0, 0, -1, 0, 0, 2])
# X = np.array([[1, 1, 1, 1],
#              [0, 0, 1, 1],
#              [0, 1, 0, 1]])
# V = np.array([[-0.62, 0.44, -0.91],
#              [-0.81, -0.09, 0.02],
#              [0.74, -0.91, -0.60],
#              [-0.82, -0.92, 0.71],
#              [-0.26, 0.68, 0.15],
#              [0.80, -0.94, -0.83]])
# H = np.dot(V, X)
# H[H <= 0], H[H > 0] = 0, 1
# height, width = H.shape
# Y = []
# for i in range(width):
#    Y.append(np.insert(H[:, i], 0, 1))
# Y = np.array(Y).T
# Z = np.dot(w, Y)
# print("The response of the output neuron to all input patterns is:", Z)

# Tutorial 7.12, 7.13 Sparse Coding
# sparsity is measured as the count of elements that are non-zero
# y1 = np.array([1, 0, 0, 0, 1, 0, 0, 0])
# y2 = np.array([0, 0, 1, 0, 0, 0, -1, 0])
# y2 = np.array([0, 0, 0, -1, 0, 0, 0, 0])
# V = np.array([[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
#              [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]])
# x = np.array([-0.05, -0.95])
# sparsity1 = np.count_nonzero(y1)
# sparsity2 = np.count_nonzero(y2)
# print("The sparsity of these two alternatives is:", sparsity1, sparsity2)
# error1 = np.linalg.norm(x.T - np.dot(V, y1.T))
# error2 = np.linalg.norm(x.T - np.dot(V, y2.T))
# print("The reconstruction errors for these two alternatives are:", error1, error2)
# if error1 > error2:
#    print("solution 2 is the better sparse code.")
# elif error1 < error2:
#    print("solution 1 is the better sparse code.")
# else:
#    print("They have the same reconstruction error.")


# Tutorial10 Q1 K-Means Clustering
# X = np.array([[-1, 3],
#              [1, 4],
#              [0, 5],
#              [4, -1],
#              [3, 0],
#              [5, 1]])
# kmeans = KMeans(n_clusters=2).fit(X)
# print("After {} iterations, the algorithm converges.".format(kmeans.n_iter_))
# print("The assigned labels of these samples are:", kmeans.labels_)  # 输出的0对应课件中的2（Week10，P18）
# print("The cluster centres are:\n", kmeans.cluster_centers_)  # 输出的顺序和课件中的顺序相反，填答案时记得倒置一下顺序
# print(kmeans.predict())  # 使用训练后kmeans对象预测新的样本的标签

# Tutorial10 Q6 Agglomerative Clustering
# X = np.array([[-1, 3],
#              [1, 2],
#              [0, 1],
#              [4, 0],
#              [5, 4],
#              [3, 2]])
# n_clusters is the target number of clusters; linkage: ward, single, average, complete
# clustering = AgglomerativeClustering(n_clusters=3, linkage='single').fit(X)
# print("The assigned labels of these samples are:", clustering.labels_)

# Tutorial10 Q2 PCA
# a, b
# X = np.array([[4, 2, 2],
#              [0, -2, 2],
#              [2, 4, 2],
#              [-2, 0, 2]])
# pca_2d = PCA(n_components=2)
# X_2d = pca_2d.fit_transform(X)
# pca_1d = PCA(n_components=1)
# X_1d = pca_1d.fit_transform(X)
# kmeans_2d = KMeans(n_clusters=2).fit(X_2d)
# kmeans_1d = KMeans(n_clusters=2).fit(X_1d)
# new_data = np.array([[3, -2, 5], [3, -2, 5]])
# new_data_2d = pca_2d.fit_transform(new_data)
# new_data_1d = pca_1d.fit_transform(new_data)
# print("Projecting the data onto 2D plane:\n", X_2d)
# print("Projecting the data onto 1D plane:\n", X_1d)
# print("The assigned labels in the transformed 2D samples are:", kmeans_2d.labels_)
# print("The new data based on the transformed 2D samples belongs to:", kmeans_2d.predict(new_data_2d))
# print("The assigned labels in the transformed 1D samples are:", kmeans_1d.labels_)
# print("The new data based on the transformed 1D samples belong to:", kmeans_1d.predict(new_data_1d))

# Tutorial10 Q3 Competitive learning (without normalisation)
# a. Find the cluster centres for each iteration
# X = np.array([[-1.0, 3.0],
#              [1.0, 4.0],
#              [0.0, 5.0],
#              [4.0, -1.0],
#              [3.0, 0.0],
#              [5.0, 1.0]])
# m1, m2, m3 = X[0] / 2, X[2] / 2, X[4] / 2
# m = [m1, m2, m3]
# lr = 0.1
# samples = np.array([X[2], X[0], X[0], X[4], X[5]])
# for i in range(len(samples)):
#    dist = []
#    for j in range(len(m)):
#        dist.append(np.linalg.norm(samples[i] - m[j]))
#    min_index = np.argmin(dist)
#    m[min_index] += lr * (samples[i] - m[min_index])
#    print("Iteration {0}: m_{1} = {2}".format(i+1, min_index+1, m[min_index]))
# print("After {0} iterations, the cluster centres are {1}:".format(len(samples), m))


# b., c. classify the samples and the new data
# def classify(X, m):
#    new_data = np.array([0, -2])
#    dist_new = []
#    classes = []
#    for i in range(len(X)):
#        dist = []
#        for j in range(len(m)):
#            dist.append(np.linalg.norm(X[i]-m[j]))
#            dist_new.append(np.linalg.norm(new_data-m[j]))
#        classes.append(np.argmin(dist) + 1)
#    print("The classes of these samples are:", classes)
#    print("The class of the new data is:", np.argmin(dist_new) + 1)


# classify(X, m)


# Tutorial10 Q4 Basic leader follower algorithm (without normalisation)
# a. Find the cluster centres for each iteration
# m = []
# theta = 3
# lr = 0.5
# m.append(samples[0])
# for i in range(len(samples)):
#    dist = []
#    for j in range(len(m)):
#        dist.append(np.linalg.norm(samples[i] - m[j]))
#    min_index = np.argmin(dist)
#    if np.linalg.norm(samples[i] - m[min_index]) < theta:
#        m[min_index] += lr * (samples[i] - m[min_index])
#    else:
#        m.append(samples[i])
#    print("Iteration {0}: m = {1}".format(i+1, m))

# b., c. classify the samples and the new data
# classify(X, m)


# Tutorial Q5 Fuzzy K-means algorithm
