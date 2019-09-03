#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name simple_array
@Description
    
@Author LiHao
@Date 2019/7/31
"""
class Solution(object):

    @classmethod
    def twoSum(cls,nums,target):
        """
        给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，
        并返回他们的数组下标
        :param nums:
        :param target:
        :return:
        HashMap 方法
        """
        numsDict = {}
        for ix,num in enumerate(nums):
            key = target - num
            if numsDict.__contains__(key):
                return [numsDict[key],ix]
            else:
                numsDict[num] = ix
        return []

    @classmethod
    def removeDuplicates(cls,nums):
        """
        给定一个排序数组，你需要在原地删除重复出现的元素，
        使得每个元素只出现一次，返回移除后数组的新长度。
        :param nums:
        :return:
        """
        if len(nums)==0: return 0
        length = 0
        i=len(nums)-1
        # 从后往前删除元素
        while i>0:
            if nums[i]!=nums[i-1]:
                length += 1
            else:
                nums.pop(i)
            i = i-1
        length += 1
        return length

    @classmethod
    def removeElement(cls,nums,val):
        """
        给定一个数组 nums 和一个值 val，你需要原地移除
        所有数值等于 val 的元素，返回移除后数组的新长度。
        :param nums:
        :param val:
        :return:
        """
        if len(nums) < 1 : return len(nums)
        i = len(nums)-1
        while i>=0:
            if nums[i] == val:
                nums.pop(i)
            i -= 1
        return len(nums)

    @classmethod
    def searchInsert(cls,nums,target):
        """
        给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
        如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
        :param nums:
        :param target:
        :return:
        """
        """
        暴力穷举法
        ix = 0
        for i in range(len(nums)):
            if target < nums[i]:
                break
            if target == nums[i]:
                return i
            if target > nums[i]:
                ix += 1
        return ix
        """
        left = 0
        right = len(nums) - 1
        while left<=right:
            mid = int((left + right) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid -1
            if nums[mid] < target:
                left = mid + 1
        return left

    @classmethod
    def maxSubArrayForce(cls,nums):
        """
        求最大子序列
        TODO : 未完待写
        :param nums:
        :return:
        """
        maxSum = nums[0]
        tmp = maxSum
        length = len(nums)
        for i in range(1,length):
            if tmp+nums[i] > tmp:
                maxSum = max(maxSum,tmp+nums[i])
                tmp = tmp + nums[i]
            else:
                maxSum = max(maxSum,nums[i] + tmp,nums[i])
                tmp = nums[i]
        return maxSum

    @classmethod
    def plusOne(cls,digits):
        """
        对一个数组加1
        :param digits:
        :return:
        """
        add = 1
        for i in range(-1,(len(digits)+1)*-1,-1):
            if add+digits[i] > 9:
                digits[i] = 0
            else:
                digits[i] = digits[i]+1
                break
        if digits[0] == 0:
            digits.insert(0,1)
        return digits

    @classmethod
    def merge(cls,nums1,m,nums2,n):
        """
        给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
        初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
        你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
        :param nums1: 
        :param m: 
        :param nums2: 
        :param n: 
        :return: 
        """
        k = m+n-1
        while k>0 and m>0 and n>0:
            if nums1[m-1] > nums2[n-1]:
                nums1[k] = nums1[m-1]
                m = m - 1
            else:
                nums1[k] = nums2[n-1]
                n = n - 1
            k = k - 1
        if n > 0:
            nums1[:n] = nums2[:n]

    @classmethod
    def generate(cls,numRows):
        """
        给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
        :param numRows:
        :return:
        """
        if numRows<=0:
            return [[]]
        t = [[1]]
        if numRows==1:
            return t
        for i in range(2,numRows+1):
            c= [0 for j in range(i)]
            c[0] =1
            c[-1] =1
            for j in range(1,len(c)-1):
                c[j] = t[i-2][j-1] + t[i-2][j]
            t.append(c)
        return t
    @classmethod
    def maxProfit_1(cls,prices):
        """
        超出了运行时间
        :param prices:
        :return:
        """
        max_profit = 0
        for i in range(0,len(prices)-1,1):
            pin = prices[i]
            for j in range(i,len(prices),1):
                if (prices[j]-pin) > max_profit:
                    max_profit = prices[j] - pin
        return max_profit

    @classmethod
    def maxProfit_2(cls,prices):
        """
        运行时间O(n)
        给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
        如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
        注意你不能在买入股票前卖出股票。
        :param prices:
        :return:
        """
        max_profit = 0
        if len(prices)<2:return max_profit
        l = len(prices)
        i = 0
        j = 1
        while i<l and j <l:
            if prices[i] < prices[j]:
                max_profit = max(prices[j]-prices[i],max_profit)
                j += 1
            else:
                i = j
                j = i+1
        return max_profit

    @classmethod
    def maxProfit_3(cls,prices):
        """
        给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
        注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        :param prices:
        :return:
        """
        pass
