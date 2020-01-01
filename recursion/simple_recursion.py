#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name simple_recursion
@Description
    简单-递归问题
@Author LiHao
@Date 2020/1/1
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    @classmethod
    def longestUnivaluePath(cls, root):
        if root is None:
            return 0
        x = cls.longestUnivaluePath(root.left)
        y = cls.longestUnivaluePath(root.right)
        if root.left is not None and root.val == root.left.val:
            x = x + 1
        if root.right is not None and root.val == root.right.val:
            y = y + 1
        if root.right!=None and root.left!=None and root.val == root.left.val and root.val == root.right.val:
            return x + y - 1
        return max(x, y)
