#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : matrix_multiply
Describe: 
    
Author : LH
Date : 2019/12/19
"""
import math


def matrix_chain_order(p):
    """
    矩阵链相乘算法-dp
    :param p: <p0,p1,p2,...,pN>
    :return:
    """
    n = len(p) - 1
    # m为记录自底向上的每个最优相乘子结构 m[i,j]
    m = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 0
    s = [[0] * n for _ in range(n)]
    for l in range(2, n + 1, 1):  # 代表链的长度
        for i in range(0, n - l + 1, 1):  # 代表开始的第一个矩阵索引
            j = i + l - 1  # 结束的矩阵索引
            # 遍历i,j中最优的分割点 k
            for k in range(i, j, 1):
                # 计算在位置k的分割的代价
                cost = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m, s


def print_optimal_parens(s, i, j):
    """
    输出最优格式
    :param s:
    :param i:
    :param j:
    :return:
    """
    if i == j:
        return "A_{}".format(i)
    else:
        return "({}{})".format(print_optimal_parens(s, i, s[i][j]), print_optimal_parens(s, s[i][j] + 1, j))


p = [5, 10, 3, 12, 5, 50, 6]
m, s = matrix_chain_order(p)
o = print_optimal_parens(s, 0, 5)
