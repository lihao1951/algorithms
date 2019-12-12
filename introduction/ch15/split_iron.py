#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : split_iron
Describe: 
    钢条切割问题
Author : LH
Date : 2019/12/12
"""
import math


def cut_rod(p, n):
    """
    递归实现
    :param p:
    :param n:
    :return:
    """
    if n == 0:
        return 0
    q = - math.inf
    for i in range(1, n + 1, 1):
        q = max(q, p[i - 1] + cut_rod(p, n - i))
    return q


def memory_cur_rod(p, n):
    m = [-math.inf] * (n + 1)
    return memory_cur_rod_aux(p, n, m)


def memory_cur_rod_aux(p, n, m):
    if m[n] >= 0:
        return m[n]
    if n == 0:
        q = 0
    else:
        q = -math.inf
        for i in range(1, n + 1, 1):
            q = max(q, p[i - 1] + memory_cur_rod_aux(p, n - i, m))
    m[n] = q
    return q


def bottom_up_cur_rod(p, n):
    m = [-math.inf] * (n + 1)
    m[0] = 0
    for j in range(1, n + 1, 1):
        q = -math.inf
        for i in range(1, j + 1, 1):
            q = max(q, p[i - 1] + m[j - i])
        m[j] = q
    return m[n]


def memory_fib(n):
    """
    带备忘的自顶向下的dp方法
    :param n:
    :return:
    """
    m = [-math.inf] * (n + 1)
    return memory_fib_aux(n,m)

def memory_fib_aux(n,m):
    if m[n] >= 0:
        return m[n]
    if n <= 1:
        q = n
    else:
        q = memory_fib_aux(n-1,m) + memory_fib_aux(n-2,m)
    m[n] = q
    return q

def bottom_up_fib(n):
    """
    自底向上的dp方法
    O(n)时间
    :param n:
    :return:
    """
    m = [-math.inf] * (n + 1)
    m[0] = 0
    m[1] = 1
    for j in range(2,n+1,1):
        q = m[j-1] + m[j-2]
        m[j] = q
    return m[n]
