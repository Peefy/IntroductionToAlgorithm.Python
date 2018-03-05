
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
        return int(2 * i)

    def __right(self, i):
        return int(2 * i + 1)

    def __parent(self, i):
        return i // 2

    def __heapsize(self, A):
        return len(A) - 1

    def maxheapify(self, A, i):
        '''
        保持堆
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
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        A = [25,33,15,41,22,34,56]
        self.maxheapify(A, 0)
        print('MaxHeapify的一个性质:', A)
        # python src/chapter6/chapter6_2.py
        # python3 src/chapter6/chapter6_2.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_2().note()
else:
    pass
