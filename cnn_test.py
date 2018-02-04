# coding:utf-8
"""
auther: yangjunpei
date: 2018-01-31
"""

import os
import struct
import numpy as np
import time
import matplotlib.pyplot as plt
from activators import *
from self_conv2d import *
import scipy as si
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cnn_convolution import *
from cnn_down_sample import *
from cnn_full_connect import *


def load_picture(file_name):
    pic1 = Image.open("2310.png")
    pic1 = pic1.resize((16, 16))
    pic1_array = np.array(pic1)
    return pic1_array / 255.0
    # picture = Image.fromarray(np.uint8(pic1_array))
    # picture.show()


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


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images / 255.0

    m, n = images.shape
    y_softmax = np.zeros([m, 10])

    for i in np.arange(0, m, 1):
        if labels[i] == 0:
            y_softmax[i, 0] = 1
        else:
            y_softmax[i, labels[i]] = 1

    return images, y_softmax


def cnn_test(x_input, y_lable):
    cnn_covlu_1 = convolution(28, 28, 1, 5, 5, 6, 0, 1, SigmoidActivator(), 0.1)
    cnn_down_sample_1 = down_sample(6, 24, 24)
    cnn_covlu_2 = convolution(12, 12, 6, 3, 3, 16, 0, 1, SigmoidActivator(), 0.1)
    cnn_down_sample_2 = down_sample(16, 10, 10)
    cnn_full_connect = full_connect(3, 1, [400, 25, 10], 0.1, SigmoidActivator())
    cnn_full_connect.full_init_w_b()
    j_cost = []
    accury_all = []

    # 解决多组数据计算梯度的问题
    j_cost_tmp = 0
    full_w_grad = []
    full_b_grad = []
    for i in range(cnn_full_connect.w_and_b_num):
        full_w_grad.append(np.zeros((cnn_full_connect.size_of_layers[i], cnn_full_connect.size_of_layers[i+1])))
        full_b_grad.append(np.zeros((1, cnn_full_connect.size_of_layers[i+1])))

    filter_1 = []
    for i in range(cnn_covlu_1.filter_number):
        filter_1.append(Filter(cnn_covlu_1.filter_height, cnn_covlu_1.filter_width, cnn_covlu_1.channel_number))

    filter_2 = []
    for i in range(cnn_covlu_2.filter_number):
        filter_2.append(Filter(cnn_covlu_2.filter_height, cnn_covlu_2.filter_width, cnn_covlu_2.channel_number))

    k = 0
    numbers_every = 100.0
    result = np.zeros((5000, 10))
    b = range(5000)

    a = range(10000)
    # c = range(5000)
    # a = a + b
    # np.random.shuffle(a)
    for j in a:
        k = k + 1
        cnn_covlu_1.conv_forward(x_input[j].reshape((1, 28, 28)))
        # print("conv1", cnn_covlu_1.output_array[0])
        cnn_down_sample_1.down_forward(cnn_covlu_1.output_array)
        cnn_covlu_2.conv_forward(cnn_down_sample_1.out_array)
        # print("conv2", cnn_covlu_2.output_array[0])
        cnn_down_sample_2.down_forward(cnn_covlu_2.output_array)

        down_layer_to_full_layer = cnn_down_sample_2.out_array.reshape((1, cnn_full_connect.size_of_layers[0]))
        cnn_full_connect.full_forward(down_layer_to_full_layer)
        # print("output_data", cnn_full_connect.output_data)

        cnn_full_connect.full_backward(y_lable[j])
        for i in range(cnn_full_connect.w_and_b_num):
            full_w_grad[i] += cnn_full_connect.w_grad[i]
            full_b_grad[i] += cnn_full_connect.b_grad[i]

        j_cost_tmp += cnn_full_connect.j_cost

        # cnn_full_connect.full_update()
        cnn_down_sample_2.down_backward(cnn_full_connect.delta[0], 0, 1)
        cnn_covlu_2.conv_backward(cnn_down_sample_2.down_delta)
        for i in range(cnn_covlu_2.filter_number):
            filter_2[i].weights_grad += cnn_covlu_2.filters[i].weights_grad
            filter_2[i].bias_grad += cnn_covlu_2.filters[i].bias_grad

        cnn_down_sample_1.down_backward(cnn_covlu_2.delta_array, cnn_covlu_2.filters, 0)
        cnn_covlu_1.conv_backward(cnn_down_sample_1.down_delta)
        for i in range(cnn_covlu_1.filter_number):
            filter_1[i].weights_grad += cnn_covlu_1.filters[i].weights_grad
            filter_1[i].bias_grad += cnn_covlu_1.filters[i].bias_grad

        if k % numbers_every == 0:
            for i in range(cnn_full_connect.w_and_b_num):
                cnn_full_connect.w_grad[i] = full_w_grad[i] / numbers_every
                cnn_full_connect.b_grad[i] = full_b_grad[i] / numbers_every
                full_w_grad[i] *= 0
                full_b_grad[i] *= 0
            for i in range(cnn_covlu_2.filter_number):
                cnn_covlu_2.filters[i].weights_grad = filter_2[i].weights_grad / numbers_every
                cnn_covlu_2.filters[i].bias_grad = filter_2[i].bias_grad / numbers_every
                filter_2[i].weights_grad *= 0
                filter_2[i].bias_grad *= 0
            for i in range(cnn_covlu_1.filter_number):
                cnn_covlu_1.filters[i].weights_grad = filter_1[i].weights_grad / numbers_every
                cnn_covlu_1.filters[i].bias_grad = filter_1[i].bias_grad / numbers_every
                filter_1[i].weights_grad *= 0
                filter_1[i].bias_grad *= 0
            cnn_full_connect.j_cost = j_cost_tmp / numbers_every
            j_cost_tmp = 0.0

            cnn_full_connect.full_update()
            cnn_covlu_2.conv_update()
            cnn_covlu_1.conv_update()
            if k % 1000 == 0:
                if cnn_full_connect.full_learn_rate > 0.00001:
                    cnn_full_connect.full_learn_rate = cnn_full_connect.full_learn_rate / 10.0
                    cnn_covlu_1.learning_rate = cnn_covlu_1.learning_rate / 10.0
                    cnn_covlu_2.learning_rate = cnn_covlu_2.learning_rate / 10.0
                    print("learning_rate", cnn_full_connect.full_learn_rate)

            print("iterator ,j_cost", k, cnn_full_connect.j_cost)
            print("cnn_full_connect.w_grad[0]", cnn_full_connect.w_grad[0][0, :])
            print("cnn_full_connect.b_grad[0]", cnn_full_connect.b_grad[0])
            # print("gradient: w", cnn_covlu_1.filters[0].weights_grad)
            # print("gradient: b", cnn_covlu_1.filters[0].bias_grad)
        # print(cnn_covlu.filters)
            j_cost.append(cnn_full_connect.j_cost)

    # accury_all = calculate_accury(cnn_covlu_1, cnn_covlu_2, cnn_down_sample_1, cnn_down_sample_2, cnn_full_connect)

    for j in b:
        cnn_covlu_1.conv_forward(x_input[j].reshape((1, 28, 28)))
        cnn_down_sample_1.down_forward(cnn_covlu_1.output_array)
        cnn_covlu_2.conv_forward(cnn_down_sample_1.out_array)
        cnn_down_sample_2.down_forward(cnn_covlu_2.output_array)

        down_layer_to_full_layer = cnn_down_sample_2.out_array.reshape((1, cnn_full_connect.size_of_layers[0]))
        cnn_full_connect.full_forward(down_layer_to_full_layer)
        result[j] = cnn_full_connect.output_data

    result = result.argmax(axis=1)
    y = y_lable[0:5000, :].argmax(axis=1)
    accury = (result == y)
    accury_all = (accury.sum()) * 1.0 / 5000
    print("accury_all", accury_all)

    plt.plot(range(len(j_cost)), j_cost)
    plt.show()

if __name__ == "__main__":
    # x_input = load_picture("2310.png")
    train_data, train_lable = load_mnist(".\data", kind="train")
    # print(train_data[0, :])
    # print("lable", train_lable[0, :])
    # print(train_data.shape, train_lable.shape)
    # print(x_train[0, :])
    # x_input = x_input.reshape((1, 16, 16))
    cnn_test(train_data, train_lable)

