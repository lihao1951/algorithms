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
            while left and len(stack_left)!=0:
                node = stack_left.pop()
                line.append(node.val)
                if node.left is not None:
                    stack_right.append(node.left)
                if node.right is not None:
                    stack_right.append(node.right)

            while not left and len(stack_right)!=0:
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
        while cur != None or len(stack)!=0:
            if cur!= None:
                stack.append(cur)
                result.append(cur.val)
                cur = cur.left
            else:
                cur = stack.pop()
                cur = cur.right
        return result

    @classmethod
    def evalRPN(cls,tokens):
        """
        150. 根据逆波兰表示法，求表达式的值。
        :param tokens:
        :return:
        """
        stack = []
        for x in tokens:
            if x in ['+','-','*','/']:
                a = stack.pop()
                b = stack.pop()
                if x == '+':
                    stack.append(a+b)
                elif x == '-':
                    stack.append(b-a)
                elif x == '*':
                    stack.append(a*b)
                else:
                    stack.append(int(b/a))
            else:
                stack.append(int(x))
        return stack.pop()

    @classmethod
    def asteroidCollision(cls,asteroids):
        """
        给定一个整数数组 asteroids，表示在同一行的行星。
        对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。
        找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

        :param asteroids:
        :return:
        """
        ...

    @classmethod
    def dailyTemperatures(cls,T):
        """
        根据每日 气温 列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高超过该日的天数。
        如果之后都不会升高，请在该位置用 0 来代替。
        例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，
        你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
        提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数
        :param T:
        :return:
        """

        ...
