#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name simple_array
@Description
    数组
        简单难度
@Author LiHao
@Date 2019/7/31
"""
class Solution(object):

    @classmethod
    def two_sum(cls,nums,target):
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
    def remove_duplicates(cls,nums):
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
    def remove_element(cls,nums,val):
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
    def search_insert(cls,nums,target):
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
    def max_sub_array_force(cls,nums):
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
    def plus_one(cls,digits):
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
    def get_row(cls,rowIndex):
        if rowIndex < 1:
            return [1]
        result = [0 for i in range(rowIndex+1)]
        result[0] = 1
        for i in range(1,rowIndex+1,1):
            # 记录上一行前一个位置的数据值
            tmp = result[0]
            for j in range(i+1):
                if j-1<0 or j>=rowIndex:
                    result[j] = tmp
                else:
                    p = tmp + result[j]
                    tmp = result[j]
                    result[j] = p
        return result[:rowIndex+1]

    @classmethod
    def max_profit_1(cls,prices):
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
    def max_profit_2(cls,prices):
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
    def max_profit_3(cls,prices):
        """
        给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
        注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        把连续的买卖看作每天都判断买不买卖，如果第二天有效益，就买卖，否则不买
        :param prices:
        :return:
        """
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0: profit += tmp
        return profit

    @classmethod
    def two_sum_1(cls,numbers,target):
        """
        给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
        函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
        说明:
            返回的下标值（index1 和 index2）不是从零开始的。
            你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
        :param numbers:
        :param target:
        :return:
        """
        if len(numbers)==0 or target<numbers[0]:
            return []
        allnum = {}
        for i in range(len(numbers)):
            sub = target - numbers[i]
            if not allnum.keys().__contains__(sub):
                allnum[numbers[i]] = i+1
            else:
                return [allnum[sub],i+1]
        return []

    @classmethod
    def two_sum_2(cls,numbers,target):
        """
        给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
        函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
        说明:
            返回的下标值（index1 和 index2）不是从零开始的。
            你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
        :param numbers:
        :param target:
        :return:
        """
        tail = len(numbers) - 1
        head = 0
        while tail > head:
            sum = numbers[head] + numbers[tail]
            if sum == target:
                return [head+1,tail+1]
            elif sum>target:
                tail -= 1
            else:
                head += 1
        return []

    @classmethod
    def majority_element(cls,nums):
        """
        给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
        你可以假设数组是非空的，并且给定的数组总是存在众数
        :param nums:
        :return:
        """
        hl = len(nums) / 2
        allnums = {}
        if len(nums) == 1:
            return nums[0]
        for i in range(len(nums)):
            if allnums.keys().__contains__(nums[i]):
                if allnums[nums[i]]+1 >= hl:
                    return nums[i]
                else:
                    allnums[nums[i]] += 1
            else:
                allnums[nums[i]] = 1

    @classmethod
    def rotate_1(cls, nums, k):
        """
        翻转字符串第K个字符
        所以此题只需要采取三次翻转的方式就可以得到目标数组，首先翻转分界线前后数组，再整体翻转一次即可。
        :param nums:
        :param k:
        :return:
        """
        for i in range(k):
            p = nums[-1]
            j = len(nums)-1
            while j > 0:
                nums[j] = nums[j-1]
                j -= 1
            nums[0] = p

    @classmethod
    def change(cls,nums,start,end):
        while start < end:
            tmp = nums[start]
            nums[start] = nums[end]
            nums[end] = tmp
            start += 1
            end -= 1

    @classmethod
    def rotate_2(cls, nums, k):
        """
        翻转字符串第K个字符
        所以此题只需要采取三次翻转的方式就可以得到目标数组，首先翻转分界线前后数组，再整体翻转一次即可。
        :param nums:
        :param k:
        :return:
        """
        l = len(nums) - 1
        k %= l+1 # 此操作将省去多余循环的移动
        cls.change(nums, 0, l-k)
        cls.change(nums, l-k+1, l)
        cls.change(nums, 0, l)

    @classmethod
    def contains_duplicate_1(cls,nums):
        """
        给定一个整数数组，判断是否存在重复元素。
        如果任何值在数组中出现至少两次，函数返回 true
        如果数组中每个元素都不相同，则返回 false
        （1）排序法
        （2）字典法
        :param nums:
        :return:
        """
        a = sorted(nums)
        i = 0
        while i < len(nums)-1:
            if a[i] == a[i+1]:
                return True
            i += 1
        return False

    @classmethod
    def contains_duplicate_2(cls, nums, k):
        """
        给定一个整数数组和一个整数k，判断数组中是否存在两个不同的索引i和j，
        使得nums[i]=nums[j]，并且i和j的差的绝对值最大为k
        (1)字典法
        :param nums:
        :return:
        """
        num_map = {}
        i = 0
        while i < len(nums):
            if num_map.keys().__contains__(nums[i]):
                if i - num_map[nums[i]] <= k:
                    return True
                else:
                    num_map[nums[i]] = i
            else:
                num_map[nums[i]] = i
            i += 1
        return False

    @classmethod
    def shortest_distance(cls, words, word1, word2):
        """
        Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
        For example,
        Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
        Given word1 = “coding”, word2 = “practice”, return 3.
        Given word1 = "makes", word2 = "coding", return 1.
        Note:
        You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.
        :param words:
        :param word1:
        :param word2:
        :return:
        """
        l = len(words)
        import math
        min_distance = math.inf
        ix1 = -1
        ix2 = -1
        for i in range(l):
            word = words[i]
            if word == word1:
                ix1 = i
            if word == word2:
                ix2 = i
            if ix1 != -1 and ix2 != -1:
                min_distance = min(min_distance, abs(ix2-ix1))
        return min_distance

    @classmethod
    def missing_number_1(cls,nums):
        """
        给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
        你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现
        (1)排序 missing_number_1
        (2)数学计算 missing_number_2
        :param nums:
        :return:
        """
        l = len(nums)
        # 先排序
        nums = sorted(nums)
        # 在线性比对
        for i in range(l):
            if i!= nums[i]:
                return i
        return l

    @classmethod
    def missing_number_2(cls,nums):
        """
        (2)数学计算
        :param nums:
        :return:
        """
        l = len(nums)
        sum_all = sum([i for i in range(l+1)])
        sum_sub = sum(nums)
        return sum_all-sum_sub

    @classmethod
    def move_zeroes(cls,nums):
        """
        283.给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
        必须在原数组上操作，不能拷贝额外的数组。
        尽量减少操作次数
        将非零的数按序交换至j所在的位置，直到遍历完所有元素，那么0就会自动排入后面
        :param nums:
        :return:
        """
        j = -1
        for i in range(len(nums)):
            if nums[i] != 0:
                j += 1
                if i != j:
                    tmp = nums[i]
                    nums[i] = nums[j]
                    nums[j] = tmp

    @classmethod
    def find_disappeared_numbers(cls,nums):
        """
        448. 找到所有数组中消失的数字
        给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
        找到所有在 [1, n] 范围之间没有出现在数组中的数字。
        您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
        :param nums:
        :return:
        """
        for i in range(len(nums)):
            k = (nums[i]-1) % len(nums)
            nums[k] += len(nums)
        print(nums)
        r = []
        for i in range(len(nums)):
            if nums[i] <= len(nums):
                r.append(i+1)
        return r
nums = [4,3,2,7,8,2,3,1]
s = Solution.find_disappeared_numbers(nums)
print(s)
