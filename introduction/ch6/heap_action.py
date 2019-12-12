#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : heap_action
Describe: 
    
Author : LH
Date : 2019/12/10
"""


def exchange(A, i, j):
    if i == j: return
    A[i] = A[i] + A[j]
    A[j] = A[i] - A[j]
    A[i] = A[i] - A[j]


def max_heapify(A, i, heap_size):
    """
    调整为最大堆
    :param A:
    :param i:
    :param heap_size: 堆的大小范围
    :return:
    """
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i
    if left < heap_size and A[largest] < A[left]:
        largest = left
    if right < heap_size and A[largest] < A[right]:
        largest = right
    if largest != i:
        exchange(A, i, largest)
        max_heapify(A, largest, heap_size)


def build_max_heap(A):
    """
    建立最大堆
    :param A:
    :return:
    """
    if not isinstance(A, list):
        raise TypeError('A must be list')
    l = len(A)
    for i in range(int((l - 1) / 2), -1, -1):
        max_heapify(A, i, l)


def heap_sort(A):
    build_max_heap(A)
    heap_size = len(A)
    for i in range(len(A) - 1, 0, -1):
        exchange(A, 0, i)
        heap_size -= 1
        max_heapify(A, 0, heap_size)


A = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
heap_sort(A)
print(A)
