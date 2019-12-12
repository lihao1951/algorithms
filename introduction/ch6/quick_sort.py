#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : quick_sort
Describe: 
    
Author : LH
Date : 2019/12/10
"""


def exchange(A, i, j):
    if i == j: return
    A[i] = A[i] + A[j]
    A[j] = A[i] - A[j]
    A[i] = A[i] - A[j]


def quick_sort(A, start, end):
    if start <= end:
        q = partition(A, start, end)
        quick_sort(A, start, q - 1)
        quick_sort(A, q + 1, end)


def partition(A, start, end):
    tmp = A[end]
    p = start - 1
    for q in range(start, end, 1):
        if A[q] <= tmp:
            p += 1
            exchange(A, p, q)
    exchange(A, p + 1, end)
    return p + 1


A = [2, 4, 8, 2, 1, 17, 5]
quick_sort(A, 0, len(A) - 1)
print(A)
