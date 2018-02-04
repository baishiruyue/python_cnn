# coding:utf-8
"""
author: yangjp
date: 2018-01-24
"""

import numpy as np
import time
from self_conv2d import *


def max_poll_one(out_array, in_array, in_array_ac):
    (height, width) = out_array.shape
    for i in range(height):
        for j in range(width):
            i_in = 2 * i
            j_in = 2 * j
            out_array[i, j] = max(in_array[i_in, j_in], in_array[i_in, j_in + 1],
                                  in_array[i_in + 1, j_in], in_array[i_in + 1, j_in + 1])
            if out_array[i, j] == in_array[i_in, j_in]:
                in_array_ac[i_in, j_in] = 1
            elif out_array[i, j] == in_array[i_in, j_in + 1]:
                in_array_ac[i_in, j_in + 1] = 1
            elif out_array[i, j] == in_array[i_in + 1, j_in]:
                in_array_ac[i_in + 1, j_in] = 1
            else:
                in_array_ac[i_in + 1, j_in + 1] = 1


class down_sample(object):
    def __init__(self, in_size, in_height, in_width):
        self.in_size = in_size
        self.in_width = in_width
        self.in_height = in_height
        self.out_size = self.in_size
        self.out_width = self.in_width / 2
        self.out_height = self.in_height / 2
        self.out_array = np.zeros((self.out_size, self.out_height, self.out_width))
        self.down_delta = np.zeros((self.in_size, self.in_height, self.in_width))

    def down_forward(self, in_array):
        self.in_array = in_array
        # 储存max_poll max(a1,a2,a3,a4) a1,a2,a3,a4的导数
        self.in_array_ac = np.zeros((self.in_size, self.in_height, self.in_width))
        for i in range(self.out_size):
            max_poll_one(self.out_array[i], self.in_array[i], self.in_array_ac[i])

    def down_backward(self, next_layer_delta, next_layer_f, is_connect_full):
        self.is_connect_full = is_connect_full
        if self.is_connect_full == 1:
            tmp = next_layer_delta.reshape(self.out_size, self.out_height, self.out_width)
            for i in range(self.out_size):
                self.down_delta[i] = np.kron(tmp[i], np.array(([1, 1], [1, 1]))) * self.in_array_ac[i]

        elif self.is_connect_full == 0:
            self.get_down_delta(next_layer_delta, next_layer_f)
        else:
            print("is_connect_full have wrong value:%d" % (self.is_connect_full))

    def get_down_delta(self, next_layer_delta, next_layer_f):
        filter_size = len(next_layer_f)
        filter_depth = next_layer_f[0].weights.shape[0]
        tmp = np.zeros((filter_depth, self.out_height, self.out_width))

        for i in range(filter_size):
            for j in range(filter_depth):
                next_layer_f_tmp = np.rot90(next_layer_f[i].weights[j], 2)
                # print("i:%d, j:%d" % (i, j))
                # print(tmp[j])
                tmp[j] += self_conv2d(next_layer_delta[i], next_layer_f_tmp, "full")
                # print(tmp[j])
        for i in range(self.out_size):
            self.down_delta[i] = np.kron(tmp[i], np.array(([1, 1], [1, 1]))) * self.in_array_ac[i]


if __name__ == "__main__":
    a = np.arange(128).reshape(2, 8, 8)
    filter_t = []
    for i in range(3):
        filter_t.append(np.arange(i*5, i*5 + 18, 1).reshape(2, 3, 3))
    # print(filter_t)
    down_sample_t = down_sample(2, 8, 8)
    down_sample_t.down_forward(a)
    down_sample_t.down_backward(np.arange(12).reshape(3, 2, 2), filter_t, 0)
    print(down_sample_t.down_delta)






