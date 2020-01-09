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
        """
        中序遍历-迭代
        :param root:
        :return:
        """
        if root is None:
            return None
        result = []
        stack = []
        cur = root
        while cur != None or len(stack) > 0:
            if cur is not None:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                result.append(cur.val)
                cur = cur.right
        return result

    @classmethod
    def zigzagLevelOrder(cls, root: TreeNode):
        """
        103. 二叉树的锯齿形层次遍历
        锯齿形访问二叉树节点
        交替使用两个栈解决问题
        :param root:
        :return:
        """
        left = True
        stack_left = []
        stack_right = []
        result = []
        if root is None:
            return result
        stack_left.append(root)
        while len(stack_left) != 0 or len(stack_right) != 0:
            line = []
            while left and len(stack_left) != 0:
                node = stack_left.pop()
                line.append(node.val)
                if node.left is not None:
                    stack_right.append(node.left)
                if node.right is not None:
                    stack_right.append(node.right)

            while not left and len(stack_right) != 0:
                node = stack_right.pop()
                line.append(node.val)
                if node.right is not None:
                    stack_left.append(node.right)
                if node.left is not None:
                    stack_left.append(node.left)
            left = not left
            result.append(line)
        return result

    @classmethod
    def preorderTraversal(cls, root: TreeNode):
        if root is None:
            return None
        result = []
        stack = []
        cur = root
        while cur != None or len(stack) != 0:
            if cur != None:
                stack.append(cur)
                result.append(cur.val)
                cur = cur.left
            else:
                cur = stack.pop()
                cur = cur.right
        return result

    @classmethod
    def evalRPN(cls, tokens):
        """
        150. 根据逆波兰表示法，求表达式的值。
        :param tokens:
        :return:
        """
        stack = []
        for x in tokens:
            if x in ['+', '-', '*', '/']:
                a = stack.pop()
                b = stack.pop()
                if x == '+':
                    stack.append(a + b)
                elif x == '-':
                    stack.append(b - a)
                elif x == '*':
                    stack.append(a * b)
                else:
                    stack.append(int(b / a))
            else:
                stack.append(int(x))
        return stack.pop()

    @classmethod
    def removeKdigits(cls, num: str, k: int):
        """
        402. 移掉K位数字
        给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
        :param num:
        :param k:
        :return:
        """
        num_stack = []
        for digit in num:
            while k and num_stack and num_stack[-1] > digit:
                num_stack.pop()
                k -= 1
                num_stack.append(digit)
        final_stack = num_stack[:-k] if k else num_stack
        return "".join(final_stack).lstrip('0') or "0"

    @classmethod
    def find132pattern(cls, nums):
        """
        456. 132模式
        给定一个整数序列：a1, a2, ..., an，一个132模式的子序列 ai, aj, ak 被定义为：
        当 i < j < k 时，ai < ak < aj。设计一个算法，当给定有 n 个数字的序列时，验证这个序列中是否含有132模式的子序列。
        :param nums:
        :return:
        """
        ...

    @classmethod
    def asteroidCollision(cls, asteroids):
        """
        行星碰撞
        :param asteroids:
        :return:
        """
        run_stack = []
        result = []
        for num in asteroids:
            if num < 0 and not run_stack:
                result.append(num)
                continue
            if num > 0:
                run_stack.append(num)
            else:
                while run_stack and num < 0 and run_stack[-1] > 0:
                    pnum = run_stack.pop()
                    if num + pnum > 0:
                        run_stack.append(pnum)
                        num = pnum
                    elif num + pnum == 0:
                        num = 0
                if num < 0:
                    run_stack.append(num)
        result.extend(run_stack)
        return result