import numpy as np
from sklearn import datasets


def sequential_delta(w, lr, t, x_k, epoch=2):
    n = 0
    for i in range(epoch):
        y = []
        for j in range(len(x_k)):
            temp = np.dot(w, np.insert(x_k[j].T, 0, 1))
            if temp > 0:
                y.append(1)
            elif temp == 0:
                y.append(0.5)
            else:
                y.append(0)
            w = w + lr * (t[j] - y[j]) * np.insert(x_k[j], 0, 1)
            n += 1
            print("n = {}, y = {}, w = {}".format(n, y[j], w))
    # print("the value of w after learning is: {}".format(w))
    # return w


# dichotomizer; iris数据集中标签为0，对应1，否则对应0
def change_label(input):
    res = []
    for idx, label in enumerate(input):
        if label == 0:
            res.append(1)
        else:
            res.append(0)
    return res


def compute_percentage(w, t, x_k):
    count = 0
    for i in range(len(x_k)):
        temp = np.dot(w, np.insert(x_k[i].T, 0, 1))
        if temp > 0 and t[i] == 1:
            count += 1
        elif temp < 0 and t[i] == 0:
            count += 1
    return count / len(x_k)


if __name__ == '__main__':
    W = np.array([5.5, -7.5, 0.5])
    rate = 1.0
    t_k = np.array([1, 1, 1, 0, 0, 0])
    x = np.array([[0.0, 2.0],
                  [1.0, 2.0],
                  [2.0, 1.0],
                  [-3.0, 1.0],
                  [-2.0, -1.0],
                  [-3.0, -2.0]])
    # times = 3
    sequential_delta(W, rate, t_k, x)

    # iris = datasets.load_iris()
    # W = np.array([-0.5, 3.5, -2.5, -2.5, 0.5])
    # rate = 0.10
    # t_k = iris.target
    # labels = change_label(t_k)
    # x = iris.data
    # w_final = sequential_delta(W, rate, labels, x)
    # proportion_before = compute_percentage(W, labels, x)
    # proportion_after = compute_percentage(w_final, labels, x)
    # print("The percentage before learning is: {}".format(proportion_before))
    # print("The percentage after learning is: {}".format(proportion_after))



