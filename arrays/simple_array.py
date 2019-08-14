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



nums = [1]
print(Solution.searchInsert(nums,7))