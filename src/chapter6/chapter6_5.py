
# python src/chapter6/chapter6_5.py
# python3 src/chapter6/chapter6_5.py

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
    
class Chapter6_5:
    '''
    CLRS 第六章 6.5 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_5().note()
        '''
        print('6.5 优先级队列')
        print('虽然堆排序算法是一个很漂亮的算法，快速排序(第7章将要介绍)的一个好的实现往往优于堆排序')
        print('堆数据结构还是有着很大的用处，一个很常见的应用：作为高效的优先级队列')
        print('和堆一样，队列也有两种，最大优先级队列和最小优先级队列')
        print('优先级队列是一种用来维护由一组元素构成的集合S的数据结构，这一组元素中的每一个都有一个关键字key')
        print('一个最大优先级队列支持一下操作')
        print('INSERT(S, x):把元素x插入集合S，这一操作可写为S<-S')
        # python src/chapter6/chapter6_5.py
        # python3 src/chapter6/chapter6_5.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_5().note()
else:
    pass
