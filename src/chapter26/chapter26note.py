# coding:utf-8
# usr/bin/python3
# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
'''

Class Chapter26_1

Class Chapter26_2

Class Chapter26_3

Class Chapter26_4

Class Chapter26_5

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

class Chapter26_1:
    '''
    chpater26.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.1 note

        Example
        ====
        ```python
        Chapter26_1().note()
        ```
        '''
        print('chapter26.1 note as follow')  
        print('第26章 最大流')
        print('为了求从一点到另一点的最短路径，可以把公路地图模型化为有向图')
        print('可以把一个有向图理解为一个流网络,并运用它来回答有关物流方面的问题')
        print('设想某物质从产生它的源点经过一个系统,流向消耗该物质的汇点(sink)这样一种过程')
        print('源点以固定速度产生该物质,而汇点则用同样的速度消耗该物质.',
            '从直观上看,系统中任何一点的物质的流为该物质在系统中运行的速度')
        print('最大流问题是关于流网络的最简单的问题')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_2:
    '''
    chpater26.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.2 note

        Example
        ====
        ```python
        Chapter26_2().note()
        ```
        '''
        print('chapter26.2 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_3:
    '''
    chpater26.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.3 note

        Example
        ====
        ```python
        Chapter26_3().note()
        ```
        '''
        print('chapter26.3 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_4:
    '''
    chpater26.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.4 note

        Example
        ====
        ```python
        Chapter26_4().note()
        ```
        '''
        print('chapter26.4 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_5:
    '''
    chpater26.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.5 note

        Example
        ====
        ```python
        Chapter26_5().note()
        ```
        '''
        print('chapter26.5 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

chapter26_1 = Chapter26_1()
chapter26_2 = Chapter26_2()
chapter26_3 = Chapter26_3()
chapter26_4 = Chapter26_4()
chapter26_5 = Chapter26_5()

def printchapter26note():
    '''
    print chapter26 note.
    '''
    print('Run main : single chapter twenty-six!')  
    chapter26_1.note()
    chapter26_2.note()
    chapter26_3.note()
    chapter26_4.note()
    chapter26_5.note()

# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
if __name__ == '__main__':  
    printchapter26note()
else:
    pass
