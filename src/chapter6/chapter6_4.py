
# python src/chapter6/chapter6_4.py
# python3 src/chapter6/chapter6_4.py
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
    
class Chapter6_4:
    '''
    CLRS 第六章 6.4 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_4().note()
        '''
        print('6.4 堆排序算法')
        print('设n = len(A)-1,堆排序算法先用BuildMaxHeap将输入数组 A[0..n]构造成一个最大堆')
        print('因为数组中的最大元素在根A[0]，则可以通过把它与A[n]互换来达到最终正确的位置')
        print('现在如果从堆中去掉结点n(通过减小heapsize[A]),可以很容易地将A[1..n-1]建成最大堆，',
            '原来根的子女仍然是最大堆,而新的元素可能违背了最大堆的性质，这时调用MaxHeapify(A, 0)就可以保持这一个性质')
        print('堆排序算法不断重复这个过程，堆的大小由n-1一直降到2')    
        print('堆排序算法的一个举例[7, 6, 5, 4, 3, 2, 1]', heap.heapsort([1, 2, 3, 4, 5, 6, 7]))
        print('HeapSort过程的时间代价O(nlgn)')
        print('调用heap.buildmaxheap的时间为O(n),n-1次heap.maxheapify中每一次的时间代价为O(lgn)')
        A = [5 ,13, 2, 25, 7, 17, 20, 8, 4]
        print('练习6.4-1 数组', _deepcopy(A), '的heapsort过程结果为：', heap.heapsort(A))
        print('练习6.4-2 证明循环不变式的过程略')
        print('练习6.4-3 按递增排序的数组A已经是一个最大堆，buildmaxheap的时间较少，但是交换元素花费时间较多')
        print(' 若A的元素按降序排列，则buildmaxheap的花费时间较多，元素交换时间差不多')
        print('练习6.4-4 略')
        print('练习6.4-5 略')
        # python src/chapter6/chapter6_4.py
        # python3 src/chapter6/chapter6_4.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_4().note()
else:
    pass
