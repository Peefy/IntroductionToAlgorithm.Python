
# python src/chapter6/chapter6_2.py
# python3 src/chapter6/chapter6_2.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter6_2:
    '''
    CLRS 第六章 6.2 算法函数和笔记
    '''

    def __left(self, i):
        return int(2 * i + 1)

    def __right(self, i):
        return int(2 * i + 2)

    def __parent(self, i):
        return (i + 1) // 2 - 1

    def __heapsize(self, A):
        return len(A) - 1

    def maxheapify(self, A, i):
        '''
        保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
        '''
        l = self.__left(i)
        r = self.__right(i)
        largest = 0
        if  l <= self.__heapsize(A) and A[l] >= A[i]:
            largest = l
        else:
            largest = i
        if r <= self.__heapsize(A) and A[r] >= A[largest]:
            largest = r
        if largest != i:
            A[i], A[largest] = A[largest], A[i]
            self.maxheapify(A, largest)
        return A

    def maxheapify_quick(self, A, i):
        '''
        保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
        '''
        count = len(A)
        largest = count
        while largest != i:
            l = self.__left(i)
            r = self.__right(i)
            if  l <= self.__heapsize(A) and A[l] >= A[i]:
                largest = l
            else:
                largest = i
            if r <= self.__heapsize(A) and A[r] >= A[largest]:
                largest = r
            if largest != i:
                A[i], A[largest] = A[largest], A[i]
                i, largest = largest, count
        return A

    def minheapify(self, A, i):
        '''
        保持堆:使某一个结点i成为最小堆(其子树本身已经为最小堆)
        '''
        l = self.__left(i)
        r = self.__right(i)
        minest = 0
        if  l <= self.__heapsize(A) and A[l] <= A[i]:
            minest = l
        else:
            minest = i
        if r <= self.__heapsize(A) and A[r] <= A[minest]:
            minest = r
        if minest != i:
            A[i], A[minest] = A[minest], A[i]
            self.minheapify(A, minest)
        return A

    def note(self):
        '''
        Summary
        =
        Print chapter6.2 note

        Example
        =
        >>> Chapter6_2().note()
        '''
        print('6.2 保持堆的性质')
        print('Max-heapify是对最大堆操作重要的子程序')
        print('其输入是一个数组A和一个下标i')
        print('假定以Left(i)和Right(i)为根的两颗二叉树都是最大堆，但是这是A[i]可能小于其子女，这样就违反了最大堆的性质')
        print('Max-heapify的过程就是使A[i]下降，使以i为根的子树成为最大堆')
        print('在算法Max-heapify的每一步里，从元素A[i], A[Left[i]], A[Right[i]]找出最大的的下标索引并存在largest中')
        print('如果A[i]已经是最大的，则以i为根的子树已经是最大堆')
        print('以该结点为根的子树又有可能违反最大堆性质，因而又要对该子树递归调用Max-heapify')
        print('Max-heapify的运行时间T(n)<=T(2n/3)+Θ(1)')
        print('根据主定理，该递归式的解为T(n)=O(lgn)')
        print('或者说，Max-heapify作用于一个高度为h的结点所需要的运行时间为O(h)')
        A = [25, 33, 15, 26, 22, 14, 16]
        self.maxheapify(A, 0)
        print('MaxHeapify的一个举例[25,33,15,26,22,14,16]，树的高度为3:', A)
        A = [27, 17, 3, 16, 13, 10, 1, 5, 7, 12, 4,8, 9, 0]
        print('练习6.2-1：在题中索引为3的元素(pyhton中索引为2)为3，写出二叉堆后要使其成为最大堆，',
            '则3和10互换后再和8互换',
            self.maxheapify(A, 2))
        print('练习6.2-2:最小堆的一个例子[3,2,4,11,12,13,14]',
            self.minheapify([3, 2, 4, 11, 12, 13, 14], 0))
        print(' 感觉最大堆保持和最小堆没有区别啊，运行时间一致，只是不等号方向不同')
        print('练习6.2-3，没效果吧，比如[7,6,5,4,3,2,1]:',self.maxheapify([7,6,5,4,3,2,1],0))
        print('练习6.2-4,没效果吧')
        A = [25, 33, 15, 26, 22, 14, 16]
        print('练习6.2-5:[25, 33, 15, 26, 22, 14, 16],', self.maxheapify_quick(A, 0))
        print('练习6.2-6 因为二叉堆树的高度就是lgn，所以最差情况就是遍历了整个二叉堆，运行时间最差为Ω(lgn)')
        # python src/chapter6/chapter6_2.py
        # python3 src/chapter6/chapter6_2.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_2().note()
else:
    pass
