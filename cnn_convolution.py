# coding:utf-8
"""
auther: yangjunpei
date: 2018-01-18
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from activators import *
from self_conv2d import *
import scipy as si


class convolution(object):
    def __init__(self, in_height, in_width, channel_number, filter_height, filter_width, filter_number,
                 zero_padding, stride, activator, learning_rate):
        """
        :param in_height:
        :param in_width:
        :param channel_number:
        :param filter_width:
        :param filter_height:
        :param filter_number:
        :param zero_padding:
        :param stride:
        :param activator:
        :param learning_rate:
        :param is_next_input: 若前一层为输入层，则做特殊处理
        """
        self.input_width = in_width
        self.input_height = in_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = 0
        self.stride = 1
        self.output_width = self.calculate_output_size(self.input_width, filter_width, 0, 1)
        self.output_height = self.calculate_output_size(self.input_height, filter_height, 0, 1)
        self.u_value = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.output_array = self.u_value
        self.delta_array = np.zeros((self.filter_number, self.output_height, self.output_width))

        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(self.filter_height, self.filter_width, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def conv_forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        for f in range(self.filter_number):
            filter = self.filters[f]
            self.u_value[f] = self_conv2d(self.input_array, filter.weights, "valid") + filter.bias
            self.output_array[f] = self.u_value[f]
            element_wise_op(self.output_array[f], self.activator.forward)

    def conv_backward(self, next_layer_delta):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        for i in range(self.filter_number):
            tmp = self.u_value[i]
            element_wise_op(tmp, self.activator.backward)
            self.delta_array[i] = tmp * next_layer_delta[i]
            self.filters[i].bias_grad = self.delta_array[i].sum()

        for i in range(self.filter_number):
            for j in range(self.channel_number):
                # print("delta_array", self.delta_array[i])
                tmp = self_conv2d(self.input_array[j], np.rot90(self.delta_array[i], 2), "valid")
                self.filters[i].weights_grad[j] = np.rot90(tmp, 2)

    def conv_update(self):
        for i in range(self.filter_number):
            self.filters[i].update(self.learning_rate)

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1


class Filter(object):
    def __init__(self, height, width, depth):
        self.weights = np.random.rand(depth, height, width)
        self.bias = 0.1
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


# 测试前向传播和反向传播后跟新的filter结果是否正确
def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = convolution(5, 5, 3, 3, 3, 2, 0, 1, SigmoidActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.conv_forward(a)
    print "前向传播结果:", cl.output_array
    cl.conv_backward(b)
    print "反向传播后更新得到的filter1:", cl.filters[0]
    print "反向传播后更新得到的filter2:", cl.filters[1]


if __name__ == "__main__":
    test()
    # a = IdentityActivator()
    # print(a)
