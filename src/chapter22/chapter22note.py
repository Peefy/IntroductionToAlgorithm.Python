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

if __name__ == '__main__':
    import graph as _g
else: 
    from . import graph as _g

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
        print('但是，当遇到稠密图(|E|接近于|V|^2)或必须很快判别两个给定顶点是否存在连接边，通常采用邻接矩阵表示法')
        print('图G=(V,E)的邻接表表示由一个包含|V|个列表的数组Adj所组成,其中每个列表对应于V中的一个顶点')
        print('对于每一个u∈V，邻接表Adj[u]包含所有满足条件(u,v)∈E的顶点v')
        print('亦即Adj[u]包含图G中所有的顶点u相邻的顶点')
        print('如果G是一个有向图,则所有邻接表的长度之和为|E|,这是因为一条形如',
            '(u,v)的边是通过让v出现在Adj[u]中来表示的')
        print('如果G是一个无向图，则所有邻接表的长度之和为2|E|')
        print('因为如果(u,v)是一条无向边,那么u就会出现在v的邻接表中')
        print('不论是有向图还是无向图，邻接表表示法都有一个很好的特性，即它所需要的存储空间为Θ(V+E)')
        print('邻接表稍作变动，即可用来表示加权图，即每条边都有着相应权值的图')
        print('权值通常由加权函数w给出，例如设G=(V,E)是一个加权函数为w的加权图')
        print('邻接表表示法稍作修改就能支持其他多种图的变体，因而有着很强的适应性')
        print('邻接表表示法也有着潜在不足之处，即如果要确定图中边(u,v)是否存在，',
            '只能在顶点u的邻接表Adj[u]中搜索v,除此之外，没有其他更快的方法')
        print('这个不足可以通过图的邻接矩阵表示法来弥补，但要在(在渐进意义下)以占用更多的存储空间作为代价')
        # !一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的
        print('一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的')
        print('邻接矩阵是沿主对角线对称的')
        print('正如图的邻接表表示一样，邻接矩阵也可以用来表示加权图')
        print('例如，如果G=(V,E)是一个加权图，其权值函数为w，对于边(u,v)∈E,其权值w(u,v)')
        print('就可以简单地存储在邻接矩阵的第u行第v列的元素中，如果边不存在，则可以在矩阵的相应元素中存储一个None值')
        # !邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的
        print('邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的')
        print('但由于邻接矩阵简单明了,因而当图较小时,更多地采用邻接矩阵来表示')
        print('另外如果一个图不是加权的，采用邻接军阵的存储形式还有一个优越性:',
            '在存储邻接矩阵的每个元素时，可以只用一个二进制位，而不必用一个字的空间')
        print('练习22.1-1 给定一个有向图的邻接表示，计算该图中每个顶点的出度和入度都为O(V+E)')
        print(' 计算出度和入度的过程相当于将邻接链表的顶点和边遍历一遍')
        print('练习22.1-2 给出包含7个顶点的完全二叉树的邻接表表示，写出其等价的邻接矩阵表示')
        g = _g.Graph()
        g.veterxs = ['1', '2', '3', '4', '5', '6', '7']
        g.edges = [('1', '2'), ('1', '3'), ('2', '4'),
               ('2', '5'), ('3', '6'), ('3', '7')]
        print(g.getmatrix())
        print('练习22.1-3 邻接链表：对于G的每个节点i，遍历；adj,将i添加到adj中遇到的每个结点')
        print(' 时间就是遍历邻接链表的时间O(V+E)')
        print('邻接矩阵：就是求G的转置矩阵，时间为O(V^2)')
        print('练习22.1-4 给定一个多重图G=(V,E)的邻接表表示,给出一个具有O(V+E)时间的算法,',
            '计算“等价”的无向图G`(V,E`)的邻接表，其中E`包含E中所有的边,',
            '且将两个顶点之间的所有多重边用一条边代表，并去掉E中所有的环')
        print('练习22.1-5 算法运行时间都为O(V^3)')
        print('练习22.1-6 ')
        print('练习22.1-7 ')
        print('练习22.1-8 ')
        print('')
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
