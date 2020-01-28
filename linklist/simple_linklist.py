#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name simple_linklist
@Description
    
@Author LiHao
@Date 2020/1/9
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    @classmethod
    def mergeTwoLists(cls, l1: ListNode, l2: ListNode) -> ListNode:
        """
        21. 合并两个有序链表
        将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
        :param l1:
        :param l2:
        :return:
        """
        head = ListNode(None)
        c = head
        while l1 and l2:
            if l1.val < l2.val:
                c.next = l1
                l1 = l1.next
            else:
                c.next = l2
                l2 = l2.next
            c = c.next
        c.next = l1 if l1 else l2
        return head.next

    @classmethod
    def mergeTwoLists_recursion(cls, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = cls.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = cls.mergeTwoLists(l1, l2.next)
            return l2

    @classmethod
    def deleteDuplicates(cls, head: ListNode) -> ListNode:
        """
        给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
        :param head:
        :return:
        """
        if head and head.next:
            front, rear = head, head.next
            while rear:
                if front.val == rear.val:
                    rear = rear.next
                    front.next = rear
                else:
                    front = front.next
                    rear = rear.next
        return head

    @classmethod
    def hasCycle(cls, head: ListNode) -> bool:
        """
        141. 环形链表
        给定一个链表，判断链表中是否有环。
        为了表示给定链表中的环，我们使用整数 pos 来表示链表尾
        连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。
        :param head:
        :return:
        """
        if head:
            slow, fast = head, head.next
            while slow and fast:
                if slow == fast:
                    return True
                else:
                    slow = slow.next
                    if fast.next:
                        fast = fast.next.next
                    else:
                        return False
        return False

    @classmethod
    def getIntersectionNode(cls, headA: ListNode, headB: ListNode):
        def getLength(head):
            c = 0
            while head:
                c += 1
                head = head.next
            return c

        a, b = headA, headB
        alen = getLength(a)
        blen = getLength(b)
        k = alen - blen if alen > blen else blen - alen
        if alen > blen:
            for _ in range(k):
                a = a.next
        else:
            for _ in range(k):
                b = b.next
        while a and b:
            if a == b:
                return a
            else:
                a = a.next
                b = b.next
        return None

    @classmethod
    def removeElements(cls, head: ListNode, val: int) -> ListNode:
        """
        203. 移除链表元素
        :param head:
        :param val:
        :return:
        """
        prev = ListNode(-1)
        prev.next = head
        c, p = prev, prev.next
        while p:
            if p.val == val:
                p = p.next
                c.next = p
            else:
                p = p.next
                c = c.next
        return prev.next

    @classmethod
    def reverseList_recursion(cls, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        node = cls.reverseList(head.next)
        head.next.next = head
        head.next = None
        return node

    @classmethod
    def reverseList(cls, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        c, p = head, None
        while c:
            tmp = c.next
            c.next = p
            p = c
            c = tmp
        return p

    @classmethod
    def isPalindrome(cls, head: ListNode) -> bool:
        """
        234. 回文链表
        请判断一个链表是否为回文链表。
        :param head:
        :return:
        """

        def reverse(head):
            if head is None or head.next is None:
                return head
            c, p = head, None
            while c:
                tmp = c.next
                c.next = p
                p = c
                c = tmp
            return p

        fast = head
        slow = head
        # 找到中点
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        fast = reverse(slow)
        slow = head
        while fast and slow:
            if slow.val != fast.val:
                return False
            slow = slow.next
            fast = fast.next
        return True

    @classmethod
    def deleteNode(cls, node):
        """
        237. 删除链表中的节点
        请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。
        :param node:
        :return:
        """
        if node.next:
            rear = node.next.next
            node.val = node.next.val
            node.next = rear
        else:
            node = None

    @classmethod
    def middleNode(cls, head: ListNode) -> ListNode:
        """
        876. 查找中间节点
        :param head:
        :return:
        """
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow

    @classmethod
    def getDecimalValue(cls, head: ListNode) -> int:
        """
        1290. 二进制链表转整数
        给你一个单链表的引用结点head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。
        请你返回该链表所表示数字的 十进制值 。
        :param head:
        :return:
        """
        # 和十进制的计算类似只不过每位都乘以2
        cur = head
        sum = 0
        while cur:
            sum = sum * 2 + cur.val
            cur = cur.next
        return sum
