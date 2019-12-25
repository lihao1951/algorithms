#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : medium_stack
Describe: 
    
Author : LH
Date : 2019/12/25
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    @classmethod
    def simplifyPath(cls, path: str) -> str:
        """
        以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。
        在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将
        目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。
        请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。
        最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。
        :param path:
        :return:
        """
        stack = []
        data = path.split('/')
        for d in data:
            if d == '..':
                if stack:
                    stack.pop()
            elif d == '.' or d == '':
                continue
            else:
                stack.append(d)
        return '/' + '/'.join(stack)

    @classmethod
    def inorderTraversal(cls, root: TreeNode):
        if root is None:
            return None
        result = []
        stack = []
        stack.append(root)

        while stack:
            if stack[-1].left is not None:
                result.append(stack[-1].left)
            else:
                node = stack.pop()
                result.append(node.val)
                if node.right is not None:
                    stack.append(node.right)
        return result
