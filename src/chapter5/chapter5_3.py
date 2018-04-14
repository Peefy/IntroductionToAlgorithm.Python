
# python src/chapter5/chapter5_3.py
# python3 src/chapter5/chapter5_3.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_3:
    '''
    CLRS 第五章 5.3 算法函数和笔记
    '''

    def sortbykey(self, array, keys):
        '''
        根据keys的大小来排序array
        '''
        A = _deepcopy(array)
        length = len(A)
        for j in range(length):
            minIndex = j
            # 找出A中第j个到最后一个元素中的最小值
            # 仅需要在头n-1个元素上运行
            for i in range(j, length):
                if keys[i] <= keys[minIndex]:
                    minIndex = i
            # 最小元素和最前面的元素交换
            min = A[minIndex]
            A[minIndex] = A[j]
            A[j] = min
        return A

    def permute_bysorting(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_bysorting([1, 2, 3, 4])
        '''
        n = len(array)
        P = _deepcopy(array)
        for i in range(n):
            P[i] = _randint(1, n ** 3)
            _time.sleep(0.002)
        return self.sortbykey(array, P)

    def randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().randomize_inplace([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_without_identity(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_without_identity([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n - 1):
            _time.sleep(0.001)
            rand = _randint(i + 1, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_with_all(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_with_all([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            _time.sleep(0.001)
            rand = _randint(0, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array
    
    def permute_by_cyclic(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_by_cyclic([1, 2, 3, 4])
        '''
        A = _deepcopy(array)
        n = len(array)
        offset = _randint(0, n - 1)
        A = _deepcopy(array)
        for i in range(n):
            dest = i + offset
            if dest >= n:
                dest = dest - n
            A[dest] = array[i]
        return A

    def note(self):
        '''
        Summary
        =
        Print chapter5.3 note

        Example
        =
        >>> Chapter5_3().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.3 随机算法')
        print('了解输入的分布有助于分析算法平均情况行为，但是许多时候无法得到有关输入分布的信息,因而不可能进行平均情况分析')
        print('但是在这些情况下，可以考虑采用随机算法')
        print('对于诸如雇佣问题之类的问题，假设输入的所有排列都是等可能的往往是有益的，通过概率分析可以设计出随机算法')
        print('不是假设输入的一个分布，而是给定一个分布。特别地，在算法运行之前，先随机地排列应聘者，以加强所有排列都是等可能的这个特性')
        print('概率分析和随机算法的区别')
        print('应聘者是以随机顺序出现的话，则雇佣一个新的办公室助理的期望次数大约是lnn')
        print('注意这个算法是确定性的，对于任何特定的输入，雇佣一个新的办公室助理的次数时钟相同')
        print('这个次数将随输入的变化而改变，而且依赖于各种应聘者的排名')
        print('给定A=(1,2,3,4,5,6)，总是会雇佣6次新的助理，因为后来的每一个都比前一个优秀(rank值大)')
        print('给定A=(6,5,4,3,2,1)，总是只会雇佣1次新的助理')
        print('再来考虑一下先对应应聘者进行排列、再确定最佳应聘者的随机算法')
        print('此时随机发生在算法上而不是输入上')
        print('Random-Hire-Assistant')
        print(' 1. randomly permute the list of candidates')
        print(' 2. best <- 0')
        print(' 3. for i <- 1 to n')
        print(' 4.     interview candidate i')
        print(' 5.     if candidate i is better than candidate best')
        print(' 6.          then best <- i')
        print(' 7.              hire candidate i')
        print(' 凭借算法第一步的改变,建立了一个随机算法，它的性能和假设应聘者以随机次序出现所得到的结果是一致的')
        print('引理5.3 过程Random-Hire-Assistant的期望雇佣费用是O(clnn)')
        print('随机排列数组：许多随机算法通过排列给定的输入数组来使输入随机化')
        print('一个常用的方法是为数组的每个元素A[i]赋一个随机的优先级P[i],', 
            '然后依据优先级对数组A中的元素进行排序，这个过程称为PermuteBySorting')
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4, 5 ,6]))
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4, 5, 6]))
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4 ,5, 6]))
        print('引理5.4 假设所有的优先级都是唯一的，过程PermuteBySorting可以产生输入的均匀随机排列')
        print('上述算法产生和原来一样的序列的概率是1/n!,而Hire雇佣问题的输入情况可能有n!中，得证')
        print('产生随机排列的一个更好方法是原地排列给定的数列：RandomizeInPlace;复杂度O(n)')
        print('引理5.5：RandomizeInPlace算法也可以计算出一个均匀随机排列')
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4, 5 ,6]))
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4, 5, 6]))
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4 ,5, 6]))
        print('随机算法通常是解决问题的最简单也是最有效的算法')
        print('练习5.3-1 略')
        print('练习5.3-2 使用随机产生非同一排列的算法[1,2,3,4]的两个随机排列', 
            self.permute_without_identity([1, 2, 3, 4]), 
            self.permute_without_identity([4, 3, 2, 1]))
        print('练习5.3-3 使用随机产生非同一排列的算法[1,2,3,4,5]的两个随机排列', 
            self.permute_with_all([1, 2, 3, 4, 5]), 
            self.permute_with_all([5, 4, 3, 2, 1]))
        print('练习5.3-4 使用随机产生非同一排列的算法[1,2,3,4,5,6]的两个随机排列', 
            self.permute_by_cyclic([1, 2, 3, 4, 5, 6]), 
            self.permute_by_cyclic([6, 5, 4, 3, 2, 1]))
        print(' 上述算法肯定不对啊，物理意义就是把数组元素循环向右平移随机个位置，但是元素的相对位置没变')
        print('练习5.3-5 略')
        print('练习5.3-6 略')
        # python src/chapter5/chapter5_3.py
        # python3 src/chapter5/chapter5_3.py
        return self

_instance = Chapter5_3()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_3().note()
else:
    pass
