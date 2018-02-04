# coding:utf-8
"""
author: yangjp
date: 2018-01-24
"""

import numpy as np
import time


# 获取卷积区域
def get_patch(input_array, i, j, filter_height, filter_width, stride):
    '''
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        input_array_conv = input_array[start_i: start_i + filter_height, start_j: start_j + filter_width]
        return input_array_conv
    elif input_array.ndim == 3:
        input_array_conv = input_array[:, start_i: start_i + filter_height, start_j: start_j + filter_width]
        return input_array_conv


def padding(input_array, kernel_height, kernel_width, str_shape):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    输入和输出均为(m m),同理卷积核行列也相同
    '''
    if str_shape == "valid":
        return input_array
    elif str_shape == "same":
        zero_pad_h = (kernel_height - 1) / 2
        zero_pad_w = (kernel_width - 1) / 2
    elif str_shape == "full":
        zero_pad_h = kernel_height - 1
        zero_pad_w = kernel_width - 1
    else:
        print("input unknow type of str_shape")

    if input_array.ndim == 3:
        input_width = input_array.shape[2]
        input_height = input_array.shape[1]
        input_depth = input_array.shape[0]
        padded_array = np.zeros((input_depth, input_height + 2 * zero_pad_h, input_width + 2 * zero_pad_w))
        padded_array[:, zero_pad_h:zero_pad_h + input_height, zero_pad_w:zero_pad_w + input_width] = input_array
        return padded_array
    elif input_array.ndim == 2:
        input_width = input_array.shape[1]
        input_height = input_array.shape[0]
        padded_array = np.zeros((input_height + 2 * zero_pad_h, input_width + 2 * zero_pad_w))
        padded_array[zero_pad_h: zero_pad_h + input_height, zero_pad_w: zero_pad_w + input_width] = input_array
        return padded_array


# 计算卷积
def self_conv2d(input_array, kernel_array, str_shape):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    argument:
    input_array：未padding前的输入数组
    kernel_array: 未翻转的卷积核
    str_shape: 字符串，"valid" or "same" or "full"
    return: 卷积后的数组
    说明，1:卷积核size小于输入数组size 2:只支持zero_pading
    '''
    channel_number = input_array.ndim
    kernel_array_number = kernel_array.ndim
    if not channel_number == kernel_array_number:
        print("输入数组的维数和卷积核的维数不同，请检查!")

    if kernel_array.ndim == 3:
        kernel_height = kernel_array.shape[1]
        kernel_width = kernel_array.shape[2]
        input_height = input_array.shape[1]
        input_width = input_array.shape[2]
        input_depth = input_array.shape[0]
    else:
        input_depth = 1
        kernel_height, kernel_width = kernel_array.shape
        input_height, input_width = input_array.shape

    padded_array = padding(input_array, kernel_height, kernel_width, str_shape)

    if input_depth == 3:
        for i in range(input_depth):
            # print(kernel_array[i])
            kernel_array[i] = np.rot90(kernel_array[i], 2)
    else:
        kernel_array = np.rot90(kernel_array, 2)

    if str_shape == "valid":
        out_height = (input_height - kernel_height) + 1
        out_width = (input_width - kernel_width) + 1
    elif str_shape == "same":
        out_height = input_height
        out_width = input_width
    elif str_shape == "full":
        out_height = input_height + kernel_height - 1
        out_width = input_width + kernel_width - 1
    else:
        print("unknow type of str_shape")
    output_array = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            a = get_patch(padded_array, i, j, kernel_height, kernel_width, 1)
            output_array[i][j] = (a * kernel_array).sum()

    return output_array


if __name__ == "__main__":
    a = np.array(([1,2,3],[4,5,6],[7,8,9]))
    b = np.ones((4,4))
    c = self_conv2d(b, a, "same")
    print(c)