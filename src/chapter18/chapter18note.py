# coding:utf-8
# usr/bin/python3
# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
'''

Class Chapter18_1

Class Chapter18_2

Class Chapter18_3

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

class Chapter18_1:
    '''
    chpater18.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.1 note

        Example
        ====
        ```python
        Chapter18_1().note()
        ```
        '''
        print('chapter18.1 note as follow')  
        print('第五部分 高级数据结构')
        # !B树是一种被设计成专门存储在磁盘上的平衡查找树
        print('第18章 B树是一种被设计成专门存储在磁盘上的平衡查找树')
        print(' 因为磁盘的操作速度要大大慢于随机存取存储器，所以在分析B树的性能时')
        print(' 不仅要看动态集合操作花了多少计算时间，还要看执行了多少次磁盘存取操作')
        print(' 对每一种B树操作，磁盘存取的次数随B树高度的增加而增加，而各种B树操作又能使B树保持较低的高度')
        print('第19章,第20章 给出可合并堆的几种实现。')
        print(' 这种堆支持操作INSERT,MINIMUM,EXTRACT-MIN和UNION.')
        print(' UNION操作用于合并两个堆。这两章中出现的数据结构还支持DELETE和DECREASE-KEY操作')
        print('第19章中出现的二项堆结构能在O(lgn)最坏情况时间内支持以上各种操作，此处n位输入堆中的总元素数')
        print('第20章 斐波那契堆对二项堆进行了改进 操作INSERT,MINIMUM和UNION仅花O(1)的实际和平摊时间')
        print(' 操作EXTRACT-MIN和DELETE要花O(lgn)的平摊时间')
        # !渐进最快的图问题算法中，斐波那契堆是其核心部分
        print(' 操作DECREASE-KEY仅花O(1)的平摊时间')
        print('第21章 用于不想交集合的一些数据结构，由n个元素构成的全域被划分成若干动态集合')
        print(' 一个由m个操作构成的序列的运行时间为O(ma(n)),其中a(n)是一个增长的极慢的函数')
        print(' 在任何可想象的应用中，a(n)至多为4.')
        print(' 这个问题的数据结构简单，但用来证明这个时间界的平摊分析却比较复杂')
        print('其他一些高级的数据结构：')
        print(' 动态树：维护一个不相交的有根树的森林')
        print('  在动态树的一种实现中，每个操作具有O(lgn)的平摊时间界；',
            '在另一种更复杂的实现中，最坏情况时间界O(lgn).动态树常用在一些渐进最快的网络流算法中')
        print(' 伸展树：是一种二叉查找树，标准的查找树操作在其上以O(lgn)的平摊时间运行,',
            '伸展树的一个应用是简化动态树')
        print(' 持久的数据结构允许在过去版本的数据结构上做查询，甚至有时候做更新,',
            '只需很小的时空代价，就可以使链式数据结构持久化的技术')
        print('第18章 B 树')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_2:
    '''
    chpater18.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.2 note

        Example
        ====
        ```python
        Chapter18_2().note()
        ```
        '''
        print('chapter18.2 note as follow')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_3:
    '''
    chpater18.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.3 note

        Example
        ====
        ```python
        Chapter18_3().note()
        ```
        '''
        print('chapter18.3 note as follow')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

chapter18_1 = Chapter18_1()
chapter18_2 = Chapter18_2()
chapter18_3 = Chapter18_3()

def printchapter18note():
    '''
    print chapter18 note.
    '''
    print('Run main : single chapter eighteen!')  
    chapter18_1.note()
    chapter18_2.note()
    chapter18_3.note()

# python src/chapter18/chapter18note.py
# python3 src/chapter18/chapter18note.py
if __name__ == '__main__':  
    printchapter18note()
else:
    pass
