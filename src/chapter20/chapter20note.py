# coding:utf-8
# usr/bin/python3
# python src/chapter20/chapter20note.py
# python3 src/chapter20/chapter20note.py
'''

Class Chapter20_1

Class Chapter20_2

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

class Chapter20_1:
    '''
    chpater20.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter20.1 note

        Example
        ====
        ```python
        Chapter20_1().note()
        ```
        '''
        print('chapter20.1 note as follow')  
        print('第20章 斐波那契堆')
        print('二项堆可以在时间O(lgn)的最坏情况时间支持可合并堆操作INSERT,MINIMUM,',
            'EXTRACT-MIN和UNION以及操作DECREASE-KEY和DELETE')
        # !斐波那契堆不涉及删除元素的可合并堆操作仅需要O(1)的平摊时间
        print('波那契堆不涉及删除元素的可合并堆操作仅需要O(1)的平摊时间,这是斐波那契堆的好处')
        print('从理论上来看，当相对于其他操作的数目来说，EXTRACT-MIN与DELETE操作的数目较小时,斐波那契堆是很理想的')
        print(' 例如某些图问题的算法对每条边都调用一次DECRESE-KEY。对有许多边的稠密图来说，每一次DECREASE-KEY调用O(1)平摊时间加起来')
        print(' 就是对二叉或二项堆的Θ(lgn)最坏情况时间的一个很大改善')
        print(' 比如解决诸如最小生成树和寻找单源最短路径等问题的快速算法都要用到斐波那契堆')
        print('但是，从实际上看，对大多数应用来说，由于斐波那契堆的常数因子以及程序设计上的复杂性',
            '使得它不如通常的二叉(或k叉)堆合适')
        print('因此，斐波那契堆主要是具有理论上的意义')
        print('如果能设计出一种与斐波那契堆有相同的平摊时间界但又简单得多的数据结构，',
            '那么它就会有很大的实用价值了')
        # !和二项堆一样，斐波那契堆由一组树构成
        print('和二项堆一样，斐波那契堆由一组树构成，实际上，这种堆松散地基于二项堆')
        print('如果不对斐波那契堆做任何DECREASE-KEY或DELETE操作，则堆中的每棵树就和二项树一样')
        print('两种堆相比，斐波那契堆的结构比二项堆更松散一些，可以改善渐进时间界。',
            '对结构的维护工作可被延迟到方便再做')
        # !斐波那契堆也是以平摊分析为指导思想来设计数据结构的很好的例子(可以利用势能方法)
        print('斐波那契堆也是以平摊分析为指导思想来设计数据结构的很好的例子(可以利用势能方法)')
        # !和二项堆一样，斐波那契堆不能有效地支持SEARCH操作
        print('和二项堆一样，斐波那契堆不能有效地支持SEARCH操作')
        print('20.1 斐波那契堆的结构')
        # !与二项堆一样，斐波那契堆是由一组最小堆有序树构成，但堆中的树并不一定是二项树
        print('与二项堆一样，斐波那契堆是由一组最小堆有序树构成，但堆中的树并不一定是二项树')
        # !与二项堆中树都是有序的不同，斐波那契堆中的树都是有根而无序的
        print('与二项堆中树都是有序的不同，斐波那契堆中的树都是有根而无序的')
        print('每个结点x包含一个指向其父结点的指针p[x],以及一个指向其任一子女的指针child[x],',
            'x所有的子女被链接成一个环形双链表,称为x的子女表。',
            '子女表的每个孩子y有指针left[y]和right[y]分别指向其左，右兄弟')
        print('如果y结点是独子')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

class Chapter20_2:
    '''
    chpater20.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter20.2 note

        Example
        ====
        ```python
        Chapter20_2().note()
        ```
        '''
        print('chapter20.2 note as follow')  
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

chapter20_1 = Chapter20_1()
chapter20_2 = Chapter20_2()

def printchapter20note():
    '''
    print chapter20 note.
    '''
    print('Run main : single chapter twenty!')  
    chapter20_1.note()
    chapter20_2.note()

# python src/chapter20/chapter20note.py
# python3 src/chapter20/chapter20note.py
if __name__ == '__main__':  
    printchapter20note()
else:
    pass
