# coding:utf-8
"""
author: yangjp
date: 2018-01-24
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
from activators import *


def operator(a):
        return a+1


def exp_operator(a):
    return np.exp(a)


def my_softmax(in_array):
    # print("in_array", in_array)
    (a, b) = in_array.shape
    out_array = np.zeros((a, b))
    tmp_array = in_array
    # print("tmp_array_before", tmp_array)
    element_wise_op(tmp_array, exp_operator)
    # print("tmp_array", tmp_array)

    sum_array = tmp_array.sum(axis=1)
    sum_array = np.repeat(sum_array.T, b, axis=0)
    out_array = (tmp_array * 1.0) / sum_array

    # print("out_array", out_array)
    return out_array


def load_data(file_name):
    orig_data = scio.loadmat(file_name)
    x_orig_data = orig_data['X']
    y_orig_data = orig_data['y']

    m, n = y_orig_data.shape
    y_softmax = np.zeros([m, 10])

    for i in np.arange(0, m, 1):
        if y_orig_data[i] == 10:
            y_softmax[i, 0] = 1
        else:
            y_softmax[i, y_orig_data[i]] = 1
    train_number = 500
    for i in np.arange(0, 5000, 500):
        if i == 0:
            x_train_data = x_orig_data[i:train_number, :]
            x_test_data = x_orig_data[train_number:i+500, :]
            y_train_data = y_softmax[i:train_number, :]
            y_test_data = y_softmax[train_number:i+500, :]
        else:
            x_train_data = np.concatenate((x_train_data, x_orig_data[i:i+train_number, :]), axis=0)
            x_test_data = np.concatenate((x_test_data, x_orig_data[i+train_number:i+500, :]), axis=0)
            y_train_data = np.concatenate((y_train_data, y_softmax[i:i+train_number, :]), axis=0)
            y_test_data = np.concatenate((y_test_data, y_softmax[i+train_number:i+500, :]), axis=0)
    # print(y_train_data[0:10, :])
    # print(x_train_data[0:1, :])
    return x_train_data, x_test_data, y_train_data, y_test_data


def full_test():
    x_train, x_test, y_train, y_test = load_data("ex3data1.mat")
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    (picture_number, character_number) = x_train.shape
    # return
    fc = full_connect(3, picture_number, [400, 25, 10], 1.5, SigmoidActivator())
    fc.full_init_w_b()
    j_cost = []
    iterator = 1
    a = range(picture_number)
    np.random.shuffle(a)
    print("learning_rate:", fc.full_learn_rate)
    for i in range(iterator):
        for j in range(500):
            print("j:", j)
            fc.full_forward(x_train)
            # fc.gradien_check(x_train, y_train)
            fc.full_backward(y_train)
            if (j % 100 == 0) and (j > 0):
                fc.full_learn_rate = (fc.full_learn_rate * 1.0) / 2
                print("learning_rate:", fc.full_learn_rate)
            fc.full_update()
            # error = fc.w_grad[0][0, 1] - fc.derivation_theta
            # print(fc.w_grad[0][0, 1])
            # print("error", error)
            # return
            j_cost.append(fc.full_get_j_cost())
            print("j_cost(%d):%f" % (j, j_cost[j]))
    # print("step:last", j_cost[picture_number - 1])
    result = element_wise_op(np.dot(x_train, fc.w[0])+fc.b[0].repeat(picture_number, axis=0), fc.activation.forward)
    result = element_wise_op(np.dot(result, fc.w[1])+fc.b[1].repeat(picture_number, axis=0), fc.activation.forward)

    result = result.argmax(axis=1)
    y = y_train.argmax(axis=1)
    accury = (result == y)
    print("accury:", accury)
    print(accury.sum())
    print(picture_number)
    b = (accury.sum()) * 1.0 / picture_number
    print(b)

    plt.plot(range(500), j_cost)
    plt.show()
    # print(result)


class full_connect(object):
    def __init__(self, layers, input_num, size_of_layers, full_learn_rate, activation):
        self.layers = layers
        self.input_num = input_num
        self.size_of_layers = size_of_layers
        if not self.layers == len(size_of_layers):
            print("neuras networks layers is wrong")
        self.input_size = size_of_layers[0]
        self.output_size = size_of_layers[-1]
        self.output_data = np.zeros((1, self.output_size))
        self.w_and_b_num = self.layers - 1
        self.w = []
        self.b = []
        self.w_grad = []
        self.b_grad = []
        for i in range(self.w_and_b_num):
            self.w.append(np.zeros((self.size_of_layers[i], self.size_of_layers[i+1])))
            self.b.append(np.zeros((1, self.size_of_layers[i+1])))
            self.w_grad.append(np.zeros((self.size_of_layers[i], self.size_of_layers[i+1])))
            self.b_grad.append(np.zeros((1, self.size_of_layers[i+1])))
        self.u = []
        self.delta = []
        for i in range(self.layers):
            self.u.append(np.zeros((self.input_num, size_of_layers[i])))
            self.delta.append(np.zeros((self.input_num, size_of_layers[i])))
        self.full_learn_rate = full_learn_rate
        self.activation = activation
        # print("full_connect_init_over")

    def full_forward(self, xdata_input):
        # print(xdata_input)
        self.u[0] = xdata_input
        a_tmp = self.u[0]
        for i in range(self.layers-1):
            self.u[i+1] = np.dot(a_tmp, self.w[i]) + self.b[i].repeat(self.input_num, axis=0)
            a_tmp = self.u[i+1]

            # 使用softmax后，最后层的输入不能再经过sigmoid函数了
            if not i == self.layers - 2:
                element_wise_op(a_tmp, self.activation.forward)

        self.output_data = my_softmax(a_tmp)
        # print("full_forward_over")

    def full_backward(self, ydata_input):
        error = self.output_data - ydata_input
        # self.j_cost = (error * error).sum() / 2
        # 交叉熵的损失函数，一定要与二分裂的损失函数区别开，后边的部分没有了
        self.j_cost = (- ydata_input * np.log(self.output_data)).sum()
        self.j_cost = self.j_cost / self.input_num
        a_tmp = self.u[self.layers-1]
        element_wise_op(a_tmp, self.activation.backward)
        # self.delta[self.layers-1] = a_tmp * error
        self.delta[self.layers - 1] = error

        for i in np.arange(self.layers-2, -1, -1):
            if i == 0:
                self.delta[0] = np.dot(self.delta[1], self.w[0].T)
            else:
                a_tmp = self.u[i]
                element_wise_op(a_tmp, self.activation.backward)
                self.delta[i] = np.dot(self.delta[i+1], self.w[i].T) * a_tmp

        for i in range(self.w_and_b_num):
            if i == 0:
                self.w_grad[0] = np.dot(self.u[0].T, self.delta[1]) / self.input_num
                self.b_grad[0] = self.delta[1].sum(axis=0) / self.input_num
            else:
                a_tmp = self.u[i]
                element_wise_op(a_tmp, self.activation.forward)
                self.w_grad[i] = np.dot(a_tmp.T, self.delta[i+1]) / self.input_num
                self.b_grad[i] = self.delta[i+1].sum(axis=0) / self.input_num
        # print("full_backward_over")

    def full_update(self):
        for i in range(self.w_and_b_num):
            self.w[i] -= self.full_learn_rate * self.w_grad[i]
            self.b[i] -= self.full_learn_rate * self.b_grad[i]
        # print("full_updata_over")

    def gradien_check(self, xdata_input, ydata_input):
        epsion = 1e-4
        self.u[0] = xdata_input
        a_tmp = self.u[0]
        gra_b_w = []
        for i in range(self.w_and_b_num):
            gra_b_w.append(self.w[i])

        gra_b_w[0][0, 1] += epsion
        for i in range(self.layers-1):
            self.u[i+1] = np.dot(a_tmp, gra_b_w[i]) + self.b[i].repeat(self.input_num, axis=0)
            a_tmp = self.u[i+1]
            element_wise_op(a_tmp, self.activation.forward)
        self.output_data = a_tmp
        self.j_cost = (- ydata_input * np.log(self.output_data) - (1 - ydata_input) * np.log(1-self.output_data)).sum()
        self.j_cost = self.j_cost / self.input_num
        cost_theta_right = self.j_cost
        print("cost_theta_right", cost_theta_right)

        a_tmp = self.u[0]
        gra_b_w[0][0, 1] -= 2 * epsion
        for i in range(self.layers-1):
            self.u[i+1] = np.dot(a_tmp, gra_b_w[i]) + self.b[i].repeat(self.input_num, axis=0)
            a_tmp = self.u[i+1]
            element_wise_op(a_tmp, self.activation.forward)
        self.output_data = a_tmp
        self.j_cost = (- ydata_input * np.log(self.output_data) - (1 - ydata_input) * np.log(1-self.output_data)).sum()
        self.j_cost = self.j_cost / self.input_num
        cost_theta_left = self.j_cost
        print("cost_theta_left", cost_theta_left)

        self.derivation_theta = (cost_theta_right - cost_theta_left) / (2 * epsion)
        print("gradient_check,error:", self.derivation_theta)

    def full_get_weight(self):
        return self.w

    def full_get_b(self):
        return self.b

    def full_get_j_cost(self):
        return self.j_cost

    def full_init_w_b(self):
        for i in range(self.w_and_b_num):
            epsion = np.sqrt(6) / np.sqrt(self.size_of_layers[i] + self.size_of_layers[i+1])
            w_b_tmp = np.random.rand(self.size_of_layers[i] + 1, self.size_of_layers[i+1]) * 2 * epsion - epsion
            self.w[i] = w_b_tmp[1:, :]
            self.b[i] = w_b_tmp[0, :].reshape(1, self.size_of_layers[i+1])


if __name__ == "__main__":
    full_test()
    a = np.array(([0,1,2,3,4,5,6,7,8,9])).reshape(1, 10)
    b = my_softmax(a)
    print(b)


