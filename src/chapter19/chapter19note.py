# coding:utf-8
# usr/bin/python3
# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
'''

Class Chapter19_1

Class Chapter19_2

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

class Chapter19_1:
    '''
    chpater19.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter19.1 note

        Example
        ====
        ```python
        Chapter19_1().note()
        ```
        '''
        print('chapter19.1 note as follow')  
        print('第19章 二项堆')
        print('可合并堆(包括二叉堆、二项堆、斐波那契堆)的数据结构，这些数据结构支持下面五种操作')
        print('MAKE-HEAP():创建并返回一个不包含任何元素的新堆')
        print('INSERT(H,x):将结点x(其关键字域中已填入了内容)插入堆H中')
        print('MINIMUM(H):返回一个指向堆H中包含最小关键字的结点的指针')
        print('EXTRACT-MIN(H):将堆H中包含的最小关键字删除，并返回一个指向该结点的指针')
        print('UNION(H1,H2):创建并返回一个包含堆H1和H2中所有结点的新堆。同时H1和H2被这个操作\"删除\"')
        print('DECREASE-KEY(H, x, k):将新关键字值k(假定它不大于当前的关键字值)赋给堆H中的结点x')
        print('DELETE(H, x):从堆H中删除结点x')
        print('    过程       |二叉堆(最坏情况)|二项堆(最坏情况)|斐波那契堆(平摊)|')
        print(' MAKE-HEAP()   |     Θ(1)      |      Θ(1)    |     Θ(1)      |')
        print(' INSERT(H,x)   |    Θ(lgn)     |     Ω(lgn)   |     Θ(1)      |')
        print(' MINIMUM(H)    |     Θ(1)      |     Ω(lgn)   |     Θ(1)      |')
        print(' EXTRACT-MIN(H)|    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print(' UNION(H1,H2)  |     Θ(n)      |     Ω(lgn)   |     Θ(1)      |')
        print(' DECREASE-KEY  |    Θ(lgn)     |     Θ(lgn)   |     Θ(1)      |')
        print(' DELETE(H, x)  |    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print('对操作SEARCH操作的支持方面看，二叉堆、二项堆、斐波那契堆都是低效的')
        print('19.1 二项树和二项堆')
        print('19.1.1 二项树')
        # !二项树Bk是一种递归定义的树。
        print('二项树Bk是一种递归定义的树。')
        # !二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子
        print('二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子')
        print('引理19.1(二项树的性质) 二项树Bk具有以下的性质')
        print('1) 共有2^k个结点')
        print('2) 树的高度为k')
        print('3) 在深度i处恰有(k i)个结点，其中i=0,1,2,...,k')
        print('4) 根的度数为k，它大于任何其他结点的度数；',
            '并且，如果根的子女从左到右编号为k-1,k-2,...,0,子女i是子树Bi的根')
        print('推论19.2 在一棵包含n个结点的二项树中，任意结点的最大度数为lgn')
        print('19.1.2 二项堆')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

class Chapter19_2:
    '''
    chpater19.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter19.2 note

        Example
        ====
        ```python
        Chapter19_2().note()
        ```
        '''
        print('chapter19.2 note as follow')  
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

chapter19_1 = Chapter19_1()
chapter19_2 = Chapter19_2()

def printchapter19note():
    '''
    print chapter19 note.
    '''
    print('Run main : single chapter nineteen!')  
    chapter19_1.note()
    chapter19_2.note()

# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
if __name__ == '__main__':  
    printchapter19note()
else:
    pass
