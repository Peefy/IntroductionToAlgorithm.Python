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
        print('可合并堆的数据结构，这些数据结构支持下面五种操作')
        print('MAKE-HEAP():创建并返回一个不包含任何元素的新堆')
        print('INSERT(H,x):将结点x(其关键字域中已填入了内容)插入堆H中')
        print('MINIMUM(H):返回一个指向堆H中包含最小关键字的结点的指针')
        print('EXTRACT-MIN(H):将堆H中包含的最小关键字删除，并返回一个指向该结点的指针')
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
