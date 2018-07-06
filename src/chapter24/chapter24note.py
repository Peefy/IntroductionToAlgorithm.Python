# coding:utf-8
# usr/bin/python3
# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
'''

Class Chapter24_1

Class Chapter24_2

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

class Chapter24_1:
    '''
    chpater24.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.1 note

        Example
        ====
        ```python
        Chapter24_1().note()
        ```
        '''
        print('chapter24.1 note as follow')  
        print('第24章 单源最短路径')
        print('一种求最短路径的方式就是枚举出所有从芝加哥到波士顿的路线,',
            '并对每条路线的长度求和,然后选择最短的一条')
        print('在最短路径问题中,给出的是一个带权有向图G=(V,E),加权函数w:E->R为从边到实型权值的映射')
        print('路径p=<v0,v1,...,vk>的权是指其组成边的所有权值之和')    
        print('边的权值还可以被解释为其他的某种度量标准,而不一定是距离')
        print('它常常被用来表示时间,费用,罚款,损失或者任何其他沿一',
            '条路径线性积累的试图将其最小化的某个量')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_2:
    '''
    chpater24.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.2 note

        Example
        ====
        ```python
        Chapter24_2().note()
        ```
        '''
        print('chapter24.2 note as follow')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

chapter24_1 = Chapter24_1()
chapter24_2 = Chapter24_2()

def printchapter24note():
    '''
    print chapter24 note.
    '''
    print('Run main : single chapter twenty-four!')  
    chapter24_1.note()
    chapter24_2.note()

# python src/chapter24/chapter24note.py
# python3 src/chapter24/chapter24note.py
if __name__ == '__main__':  
    printchapter24note()
else:
    pass
