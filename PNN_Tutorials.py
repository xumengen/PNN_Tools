import numpy as np
# W = np.array([[1, 1, 0], [1, 1, 1]])
# x = np.array([1, 1, 0])
# n = 5
# y = np.array([[0, 0, 0], [0, 0, 0]])
# i = 1
# while i <= n:
#    e = x.T - np.dot(W.T, y)
#    y = y + 0.5 * np.dot(W, e)
#    i = i + 1
# print(y)


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
# from sklearn import datasets  # import data sets module from sklearn library
# from sklearn.neighbors import KNeighborsClassifier  # import KNN classifier model from sklearn's neighbors module
# knn = KNeighborsClassifier(n_neighbors=3)  # Define KNN classier model (n_neighbors represents the value of k)
# y = np.array([1, 2, 2, 3, 3])  # specify prediction target
# X = np.array([[0.15, 0.35],
#              [0.15, 0.28],
#              [0.12, 0.2],
#              [0.1, 0.32],
#              [0.06, 0.25]])  # choose features, y = f(X)
# knn.fit(X, y)  # fit the model (the heart of modelling)
# print(knn.predict([[0.1, 0.25]]))  # using this model to predict
