# coding:utf-8
# usr/bin/python3
# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
'''

Class Chapter23_1

Class Chapter23_2

Class Chapter23_3

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

class Chapter23_1:
    '''
    chpater23.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.1 note

        Example
        ====
        ```python
        Chapter23_1().note()
        ```
        '''
        print('chapter23.1 note as follow')  
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

class Chapter23_2:
    '''
    chpater23.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.2 note

        Example
        ====
        ```python
        Chapter23_2().note()
        ```
        '''
        print('chapter23.2 note as follow')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py


class Chapter23_3:
    '''
    chpater23.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.3 note

        Example
        ====
        ```python
        Chapter23_3().note()
        ```
        '''
        print('chapter23.3 note as follow')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

chapter23_1 = Chapter23_1()
chapter23_2 = Chapter23_2()
chapter23_3 = Chapter23_3()

def printchapter23note():
    '''
    print chapter23 note.
    '''
    print('Run main : single chapter twenty-three!')  
    chapter23_1.note()
    chapter23_2.note()
    chapter23_3.note()

# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
if __name__ == '__main__':  
    printchapter23note()
else:
    pass
