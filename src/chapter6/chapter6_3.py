
# python src/chapter6/chapter6_3.py
# python3 src/chapter6/chapter6_3.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
else:
    from . import heap

class Chapter6_3:
    '''
    CLRS 第六章 6.3 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.3 note

        Example
        =
        >>> Chapter6_3().note()
        '''
        print('6.3 建堆')
        print('可以自底向上地使用Max-heapify将一个数组A变成最大堆[0..len(A)-1]')
        print('子数组A[(n/2)+1..n]中的元素都是树中的叶子，因此每个都可以看做是只含一个元素的堆')
        print('BuildMaxHeap的运行时间的界为O(n)')
        print('一个n元素堆的高度为[lgn],并且在任意高度上，至多有[n/2^(h+1)]个结点')
        A = [5,3,17,10,84,19,6,22]
        print('练习6.3-1 BuildMaxHeap作用于数组', _deepcopy(A), 
            "的过程为：", heap.buildmaxheap(A))
        print('练习6.3-2 因为heap.maxheapify使某节点变成最大堆过程的前提是其所有的子树已经是最大堆,所以要先从子树开始，也就是循环下标从大到小')
        print('练习6.3-3 定理：在任一包含n个元素的堆中，至多有[n/2^(h+1)]个高度为h的结点')
        # python src/chapter6/chapter6_3.py
        # python3 src/chapter6/chapter6_3.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_3().note()
else:
    pass
