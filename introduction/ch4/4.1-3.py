#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : 4.1-3
Describe: 
    最大子数组 暴力求解、分治法及线性方法对比
Author : LH
Date : 2019/9/6
"""
import math
import time
import matplotlib.pyplot as plt

array = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]


def direct(array):
    """
    暴力求解法
    :param array:
    :return:
    """
    l = len(array)
    istart = 0
    iend = 0
    subsum = -float('inf')
    for i in range(l - 1):
        sum = array[i]
        for j in range(i + 1, l):
            if sum + array[j] > subsum:
                istart = i
                subsum = sum + array[j]
                iend = j
            sum += array[j]
    return istart, iend, subsum


def find_middle_maxsum(array, low, mid, high):
    """
    查找跨过mid的最大子数组
    :param array:
    :param low:
    :param mid:
    :param high:
    :return:
    """
    i = mid
    sum = 0
    left_sum = -float('inf')
    right_sum = -float('inf')
    left_start = mid - 1
    right_end = mid + 1
    while i >= low:
        if sum + array[i] > left_sum:
            left_sum = sum + array[i]
            left_start = i
        sum += array[i]
        i -= 1
    i = mid + 1
    sum = 0
    while i <= high:
        if sum + array[i] > right_sum:
            right_sum = sum + array[i]
            right_end = i
        sum += array[i]
        i += 1
    return left_start, right_end, right_sum + left_sum


def divide(array, low, high):
    """
    分治法求解
    :param array:
    :param low:
    :param high:
    :return:
    """
    if low == high:
        return low, high, array[low]
    mid = math.floor((high + low) / 2)
    # 求解左半部分
    left_low, left_high, left_sum = divide(array, low, mid)
    # 求解右半部分
    right_low, right_high, right_sum = divide(array, mid + 1, high)
    # 求解中间跨mid的部分
    middle_low, middle_high, middle_sum = find_middle_maxsum(array, low, mid, high)
    # 对比每一块的大小，返回最大的那一块
    if left_sum >= right_sum and left_sum >= middle_sum:
        return left_low, left_high, left_sum
    elif right_sum > middle_sum and middle_sum > left_sum:
        return right_low, right_high, right_sum
    else:
        return middle_low, middle_high, middle_sum


def linear(array):
    """
    线性方法
    :param array:
    :return:
    """
    l = len(array)
    maxSum = -float('inf')
    thisSum = -float('inf')
    istart = -1
    iend = -1
    for i in range(l):
        thisSum += array[i]
        if thisSum > maxSum:
            maxSum = thisSum
            iend = i
        elif thisSum < 0:
            thisSum = 0
            istart = i + 1
    return istart, iend, maxSum


def plot_run_time(array):
    array = array * 30
    l = len(array)
    x = []
    divide_y = []
    direct_y = []
    linear_y = []
    for i in range(0, l, 10):
        newArray = array[:i + 1]
        x.append(i)
        start = time.time()
        direct(newArray)
        end = time.time()
        direct_y.append((end - start) * 20)

        start = time.time()
        divide(newArray, 0, len(newArray) - 1)
        end = time.time()
        divide_y.append((end - start) * 20)

        start = time.time()
        linear(newArray)
        end = time.time()
        linear_y.append((end - start) * 20)
    plt.plot(x, direct_y, marker='*', c='r', label='direct')
    plt.plot(x, divide_y, marker='+', c='b', label='divide')
    plt.plot(x, linear_y, marker='+', c='g', label='linear')
    plt.xlabel("N")
    plt.ylabel("Time")
    plt.title("direct vs divide vs linear")
    plt.legend()
    plt.show()


plot_run_time(array)
