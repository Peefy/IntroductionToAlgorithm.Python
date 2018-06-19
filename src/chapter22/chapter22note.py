# coding:utf-8
# usr/bin/python3
# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
'''

Class Chapter22_1

Class Chapter22_2

Class Chapter22_3

Class Chapter22_4

Class Chapter22_5

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

class Chapter22_1:
    '''
    chpater22.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.1 note

        Example
        ====
        ```python
        Chapter22_1().note()
        ```
        '''
        print('chapter22.1 note as follow')  
        print('第六部分 图算法')
        print('图是计算机科学中常用的一类数据结构，有关图的算法也是计算机科学中基础性算法')
        print('有许多有趣而定计算问题都是用图来定义的')
        print('第22章介绍图在计算机中的表示，并讨论基于广度优先或深度优先图搜索的算法')
        print(' 给出两种深度优先搜索的应用；根据拓扑结构对有向无回路图进行排序，以及将有向图分解为强连通子图')
        print('第23章介绍如何求图的最小权生成树(minimum-weight spanning tree)')
        print(' 定义：即当图中的每一条边都有一个相关的权值时，',
            '这种树由连接了图中所有顶点的、且权值最小的路径所构成')
        print(' 计算最小生成树的算法是贪心算法的很好的例子')
        print('第24章和第25章考虑当图中的每条边都有一个相关的长度或者\"权重\"时，如何计算顶点之间的最短路径问题')
        print('第24章讨论如何计算从一个给定的源顶点至所有其他顶点的最短路径问题')
        print('第25章考虑每一对顶点之间最短路径的计算问题')
        print('第26章介绍在物流网络(有向图)中，物流的最大流量计算问题')
        print('在描述某一给定图G=(V, E)上的一个图算法的运行时间，通常以图中的顶点个数|V|和边数|E|来度量输入规模')
        print('比如可以讲该算法的运行时间为O(VE)')
        print('用V[G]表示一个图G的顶点集,用E[G]表示其边集')
        print('第22章 图的基本算法')
        print('22.1 图的表示')
        print('要表示一个图G=(V,E),有两种标准的方法，即邻接表和邻接矩阵')
        print('这两种表示法即可以用于有向图，也可以用于无向图')
        print('通常采用邻接表表示法，因为用这种方法表示稀疏图比较紧凑')
        print('')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py


class Chapter22_2:
    '''
    chpater22.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.2 note

        Example
        ====
        ```python
        Chapter22_2().note()
        ```
        '''
        print('chapter22.2 note as follow')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_3:
    '''
    chpater22.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.3 note

        Example
        ====
        ```python
        Chapter22_3().note()
        ```
        '''
        print('chapter22.3 note as follow')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_4:
    '''
    chpater22.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.4 note

        Example
        ====
        ```python
        Chapter22_4().note()
        ```
        '''
        print('chapter22.4 note as follow')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py


class Chapter22_5:
    '''
    chpater22.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.5 note

        Example
        ====
        ```python
        Chapter22_5().note()
        ```
        '''
        print('chapter22.5 note as follow')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

chapter22_1 = Chapter22_1()
chapter22_2 = Chapter22_2()
chapter22_3 = Chapter22_3()
chapter22_4 = Chapter22_4()
chapter22_5 = Chapter22_5()

def printchapter22note():
    '''
    print chapter22 note.
    '''
    print('Run main : single chapter twenty-two!')  
    chapter22_1.note()
    chapter22_2.note()
    chapter22_3.note()
    chapter22_4.note()
    chapter22_5.note()

# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
if __name__ == '__main__':  
    printchapter22note()
else:
    pass
