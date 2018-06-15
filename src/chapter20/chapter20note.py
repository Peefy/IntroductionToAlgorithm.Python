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
