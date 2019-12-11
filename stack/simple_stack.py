#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : simple_stack
Describe: 
    栈-简单
Author : LH
Date : 2019/12/11
"""
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
            push(x) -- 将元素 x 推入栈中。
            pop() -- 删除栈顶的元素。
            top() -- 获取栈顶元素。
            getMin() -- 检索栈中的最小元素。
        """
        self._stack = []
        self._min = []
        ...

    def push(self, x: int) -> None:
        self._stack.append(x)
        if len(self._min) == 0:
            self._min.append(x)
        else:
            self._min.append(min(self._min[-1],x))

    def pop(self) -> None:
        if len(self._stack)!=0:
            self._stack.pop(-1)
            self._min.pop(-1)

    def top(self) -> int:
        if len(self._stack) != 0:
            return self._stack[-1]
        else:
            return None

    def getMin(self) -> int:
        if len(self._min)!=0:
            return self._min[-1]
        else:
            return None


class Solution(object):
    @classmethod
    def is_valid(cls,s):
        """
        给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
        有效字符串需满足：
        左括号必须用相同类型的右括号闭合。
        左括号必须以正确的顺序闭合。
        注意空字符串可被认为是有效字符串
        :return:
        """
        label = {'{':'}','[':']','(':')'}
        stack=[]
        for i in s:
            if label.keys().__contains__(i):
                stack.append(label.get(i))
            else:
                if len(stack) == 0:
                    return False
                sl = stack.pop(-1)
                if i is not sl:
                    return False
        return len(stack) is 0

# 用队列实现栈 两个队列/循环队列
# 用栈实现队列 两个栈
