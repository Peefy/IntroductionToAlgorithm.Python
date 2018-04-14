
# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

from numpy import *
import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_4:
    '''
    CLRS 第四章 4.4 算法函数和笔记
    '''

    def __bitGet(self, number, n):
        return (((number)>>(n)) & 0x01)  

    def findBinNumberLost(self, array):
        '''
        找出所缺失的整数
        '''
        length = len(array)
        A = deepcopy(array)
        B = arange(0, length + 1, dtype=float)
        for i in range(length + 1):
            B[i] = math.inf
        for i in range(length):
            # 禁止使用A[i]
            B[A[i]] = A[i]
        for i in range(length + 1):
            if B[i] == math.inf:
                return i

    def __findNumUsingBinTreeRecursive(self, array, rootIndex, number):
        root = deepcopy(rootIndex)
        if root < 0 or root >= len(array):
            return False
        if array[root] == number:
            return True
        elif array[root] > number:
            return self.__findNumUsingBinTreeRecursive(array, root - 1, number)
        else:
            return self.__findNumUsingBinTreeRecursive(array, root + 1, number)

    def findNumUsingBinTreeRecursive(self, array, number):
        '''
        在排序好的数组中使用递归二叉查找算法找到元素

        Args
        =
        array : a array like, 待查找的数组
        number : a number, 待查找的数字

        Return
        =
        result :-> boolean, 是否找到

        Example
        =
        >>> Chapter4_4().findNumUsingBinTreeRecursive([1,2,3,4,5], 6)
        >>> False

        '''
        middle = (int)(len(array) / 2);
        return self.__findNumUsingBinTreeRecursive(array, middle, number)

    def note(self):
        '''
        Summary
        =
        Print chapter4.4 note

        Example
        =
        >>> Chapter4_4().note()
        '''
        print('4.4 主定理的证明 page45 pdf53 画递归树证明 详细过程略')
        print('思考题4-1')
        print(' a) T(n)=Θ(n)')
        print(' b) T(n)=Θ(lgn)')
        print(' c) T(n)=Θ(n^2)')
        print(' d) T(n)=Θ(n^2)')
        print(' e) T(n)=Θ(n^2)')
        print(' f) T(n)=Θ(n^0.5*lgn)')
        print(' g) T(n)=Θ(lgn)')
        print(' h) T(n)=Θ(nlgn)')
        print('思考题4-2')
        print(' 数组[0,1,2,3,5,6,7]中所缺失的整数为:',
            self.findBinNumberLost([7, 5, 2, 3, 0, 1, 6]))
        print('思考题4-3')
        print('数组[1,2,3,4,5]中是否包含6:', self.findNumUsingBinTreeRecursive([1, 2, 3, 4, 5], 6))
        print('数组[1,2,3,4,5]中是否包含2:', self.findNumUsingBinTreeRecursive([1, 2, 3, 4, 5], 2))
        print('思考题4-4 略')
        print('思考题4-5 略')
        print('思考题4-6 略')
        print('思考题4-7 略')
        
        # python src/chapter4/chapter4_4.py
        # python3 src/chapter4/chapter4_4.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_4().note()
else:
    pass
