# coding:utf-8
# usr/bin/python3
# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
'''

Class Chapter25_1

Class Chapter25_2

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

if __name__ == '__main__':
    import graph as _g
else:
    from . import graph as _g

class Chapter25_1:
    '''
    chpater25.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter25.1 note

        Example
        ====
        ```python
        Chapter25_1().note()
        ```
        '''
        print('chapter25.1 note as follow')  
        print('第25章 每对顶点间的最短路径')
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

class Chapter25_2:
    '''
    chpater25.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter25.2 note

        Example
        ====
        ```python
        Chapter25_2().note()
        ```
        '''
        print('chapter25.2 note as follow')  
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

chapter25_1 = Chapter25_1()
chapter25_2 = Chapter25_2()

def printchapter25note():
    '''
    print chapter25 note.
    '''
    print('Run main : single chapter twenty-five!')  
    chapter25_1.note()
    chapter25_2.note()

# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
if __name__ == '__main__':  
    printchapter25note()
else:
    pass
