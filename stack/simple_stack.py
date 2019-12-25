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
            self._min.append(min(self._min[-1], x))

    def pop(self) -> None:
        if len(self._stack) != 0:
            self._stack.pop(-1)
            self._min.pop(-1)

    def top(self) -> int:
        if len(self._stack) != 0:
            return self._stack[-1]
        else:
            return None

    def getMin(self) -> int:
        if len(self._min) != 0:
            return self._min[-1]
        else:
            return None


class Solution(object):
    @classmethod
    def is_valid(cls, s):
        """
        给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
        有效字符串需满足：
        左括号必须用相同类型的右括号闭合。
        左括号必须以正确的顺序闭合。
        注意空字符串可被认为是有效字符串
        :return:
        """
        label = {'{': '}', '[': ']', '(': ')'}
        stack = []
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
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._queue1 = []
        self._queue2 = []
        self._count = 0

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self._queue1.append(x)
        self._count += 1

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        while self._count > 1:
            self._queue2.append(self._queue1.pop(0))
            self._count -= 1
        result = self._queue1.pop(0)
        self._count -= 1
        while len(self._queue2) > 0:
            self._queue1.append(self._queue2.pop(0))
            self._count += 1
        return result

    def top(self) -> int:
        """
        Get the top element.
        """
        if self._count == 0:
            return 0
        return self._queue1[self._count - 1]

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        if len(self._queue1) == 0:
            return True
        return False


class Solution(object):
    @classmethod
    def nextGreaterElement(cls, nums1, nums2):
        """
        496.下一个更大元素 I
        给定两个没有重复元素的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。找到 nums1 中每个元素在 nums2 中的下一个比其大的值。
        nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出-1。
        :param nums1:
        :param nums2:
        :return:
        """
        stack, hash = [], {}
        for n in nums2:
            while stack and stack[-1] < n:
                hash[stack.pop()] = n
            stack.append(n)
        return [hash.get(x, -1) for x in nums1]

    @classmethod
    def calPoints(cls, ops) -> int:
        """
        你现在是棒球比赛记录员。
        给定一个字符串列表，每个字符串可以是以下四种类型之一：
        1.整数（一轮的得分）：直接表示您在本轮中获得的积分数。
        2. "+"（一轮的得分）：表示本轮获得的得分是前两轮有效 回合得分的总和。
        3. "D"（一轮的得分）：表示本轮获得的得分是前一轮有效 回合得分的两倍。
        4. "C"（一个操作，这不是一个回合的分数）：表示您获得的最后一个有效 回合的分数是无效的，应该被移除。

        每一轮的操作都是永久性的，可能会对前一轮和后一轮产生影响。
        你需要返回你在所有回合中得分的总和
        :param ops:
        :return:
        """
        stack = []
        for n in ops:
            if n is 'C':
                stack.pop()
            elif n is 'D':
                stack.append(stack[-1] * 2)
            elif n is '+':
                stack.append(stack[-1] + stack[-2])
            else:
                stack.append(int(n))
        return sum(stack)

    @classmethod
    def backspaceCompare(cls, S: str, T: str) -> bool:
        """
        884.比较含退格的字符串
        给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符
        :param S:
        :param T:
        :return:
        """
        return cls._output(S) == cls._output(T)

    @classmethod
    def _output(cls, s: str):
        stack = []
        for x in s:
            if x is '#':
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(x)
        return ''.join(stack)

    @classmethod
    def removeDuplicates(cls, S: str) -> str:
        """
        1047. 删除字符串中的所有相邻重复项
        给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
        在 S 上反复执行重复项删除操作，直到无法继续删除。
        在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
        :param S:
        :return:
        """
        stack = []
        for x in S:
            if stack and stack[-1] == x:
                stack.pop()
            else:
                stack.append(x)
        return ''.join(stack)


S = 'abbaca'
print(Solution.removeDuplicates(S))
