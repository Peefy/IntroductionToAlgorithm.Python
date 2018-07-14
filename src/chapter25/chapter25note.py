# coding:utf-8
# usr/bin/python3
# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
'''

Class Chapter25_1

Class Chapter25_2

Class Chapter25_3

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
    import extendshortestpath as _esp
else:
    from . import graph as _g
    from . import extendshortestpath as _esp

class Chapter25_1:
    '''
    chpater25.1 note and function
    '''
    def solve_25_1_1(self):
        g = _g.Graph()
        g.addvertex(['1', '2', '3', '4', '5', '6'])
        g.addedgewithweight('1', '5', -1, _g.DIRECTION_TO)
        g.addedgewithweight('2', '1', 1, _g.DIRECTION_TO)
        g.addedgewithweight('2', '4', 2, _g.DIRECTION_TO)
        g.addedgewithweight('3', '2', 2, _g.DIRECTION_TO)
        g.addedgewithweight('3', '6', -8, _g.DIRECTION_TO)
        g.addedgewithweight('4', '1', -4, _g.DIRECTION_TO)
        g.addedgewithweight('4', '5', 3, _g.DIRECTION_TO)
        g.addedgewithweight('5', '2', 7, _g.DIRECTION_TO)
        g.addedgewithweight('6', '2', 5, _g.DIRECTION_TO)
        g.addedgewithweight('6', '3', 10, _g.DIRECTION_TO)
        print('权值矩阵为')
        W = g.getmatrixwithweight()
        print(W)
        print('迭代最短距离矩阵为')
        L = _esp.faster_all_pairs_shortest_paths(W)
        print(L)
        print('pi矩阵为')
        print(_esp.getpimatrix(g, L, W))

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
        print('在本章中，讨论找出图中每对顶点最短路径的问题')
        print('例如，对一张公路图，需要制表说明每对城市间的距离，就可能出现这种问题')
        print('给定一加权有向图G=(V,E),其加权函数w：E->R为边到实数权值的映射',
            '对于每对顶点u,v∈V，希望找出从u到v的一条最短(最小权)路径,',
            '其中路径的权值是指其组成边的权值之和')
        print('通常希望以表格形式输出结果:第u行第v列的元素应是从u到v的最短路径的权值')
        print('可以把单源最短路径算法运行|V|次来解决每对顶点间最短路径问题,每一次运行时,',
            '轮流把每个顶点作为源点。如果所有边的权值是非负的,可以采用Dijkstra算法')
        print('如果采用线性数组来实现最小优先队列,算法的运行时间为O(V^3+VE)=O(V^3)')
        print('如果是稀疏图,采用二叉最小堆来实现最小优先队列,就可以把算法的运行时间改进为O(VElgV)')
        print('或者采用斐波那契堆来实现最小优先队列,其算法运行时间为O(V^2lgV+VE)')
        print('如果允许有负权值的边,就不能采用Dijkstra算法.必须对每个顶点运行一次速度较慢的Bellman-Ford算法',
            '它的运行时间为O(V^2E),而在稠密图上的运行时间为O(V^4)')
        print('本章中每对顶点间最短路径算法的输出是一个n*n的矩阵D=(dij)',
            '其中元素dij是从i到j的最短路径的权值。就是说，如果用d(i,j)表示从顶点i到顶点j的最短路径的权值',
            '则在算法终止时dij=d(i,j)')
        print('为了求解对输入邻接矩阵的每对顶点间最短路径问题,不仅要算出最短路径的权值,而且要计算出一个前驱矩阵∏',
            '其中若i=j或从i到j没有通路,则pi(i,j)为None,否则pi(i,j)表示从i出发的某条最短路径上j的前驱顶点')
        print('25.1节介绍一个基于矩阵乘法的动态规划算法,求解每对顶点间的最短路径问题',
            '由于采用了重复平方的技术,算法的运行时间为Θ(V^3lgV)')
        print('25.2节给出另一种动态规划算法,即Floyd-Warshall算法,该算法的运行时间为Θ(V^3)')
        print('25.2节还讨论求有向图传递闭包的问题,这一问题与每对顶点间最短路径有关系')
        print('25.3节介绍Johnson算法,Johnson算法采用图的邻接表表示法',
            '该算法求解每对顶点间最短路径问题所需的时间为O(V^2lgV+VE)',
            '对大型稀疏图来说这是一个很好的算法')
        print('25.1 最短路径与矩阵乘法')
        print('动态规划算法,用来解决有向图G=(V,E)上每对顶点间的最短路径问题')
        print('动态规划的每一次主循环都将引发一个与矩阵乘法运算十分相似的操作',
            '因此算法看上去很像是重复的矩阵乘法','开始先找到一种运行时间为Θ(V^4)的算法',
            '来解决每对顶点间的最短路径问题,然后改进这一算法,使其运行时间达到Θ(V^3lgV)')
        print('动态规划算法的几个步骤')
        print('1) 描述一个最优解的结构')
        print('2) 递归定义一个最优解的值')
        print('3) 按自底向上的方式计算一个最优解的值')
        print('最短路径最优解的结构')
        print('  对于图G=(V,E)上每对顶点间的最短路径问题,',
            '已经在引理24.1中证明了最短路径的所有子路径也是最短路径')
        print('假设图以邻接矩阵W=(wij)来表示,考察从顶点i到顶点j的一条最短路径p,假设p至多包含m条边',
            '假设图中不存在权值为负的回路,则m必是有限值.如果i=j,则路径p权值为0而且没有边')
        print('若顶点i和顶点j是不同顶点,则把路径p分解为i~k->j，其中路径p\'至多包含m-1条边')
        print('每对顶点间最短路径问题的一个递归解')
        print('  设lij(m)是从顶点i到顶点j的至多包含m条边的任何路径的权值最小值.',
            '当m=0时,从i到j存在一条不包含边的最短路径当且仅当i=j')
        print('  对m>=1,先计算lij(m-1),以及从i到j的至多包含m条边的路径的最小权值,',
            '后者是通过计算j的所有可能前趋k而得到的,然后取二者中的最小值作为lij(m),因此递归定义')
        print('  lij(m)=min(lij(m-1),min{lik(m-1)+wkj})=min{lik(m-1)+wkj}')
        print('  后一等式成立是因为对所有j,wij=0')
        print('自底向上计算最短路径的权值')
        print('  把矩阵W=(wij)作为输入,来计算一组矩阵L(1),L(2),...,L(n-1)',
            '其中对m=1,2,...,n-1,有L(m)=(lij(m)).最后矩阵L(n-1)包含实际的最短路径权值',
            '注意：对所有的顶点i,j∈V，lij(1)=wij,因此L(1)=W')
        print('算法的输入是给定矩阵L(m-1)和W,返回矩阵L(m),就是把已经计算出来的最短路径延长一条边')
        print('改进算法的运行时间')
        print('  目标并不是计算出全部的L(m)矩阵，所感兴趣的是仅仅是倒数第二个迭代矩阵L(n-1)',
            '如同传统的矩阵乘法满足结合律,EXTEND-SHORTEST-PATHS定义的矩阵乘法也一样')
        print('  通过两两集合矩阵序列,只需计算[lg(n-1)]个矩阵乘积就能计算出L(n-1)')
        print('  因为[lg(n-1)]个矩阵乘积中的每一个都需要Θ(n^3)时间',
            '因此FAST-ALL-PAIRS-SHORTEST-PATHS的运行时间Θ(n^3lgn)')
        print('  算法中的代码是紧凑的,不包含复杂的数据结构,因此隐含于Θ记号中的常数是很小的')
        _esp.test_show_all_pairs_shortest_paths()
        print('练习25.1-1 代码如下')
        self.solve_25_1_1()
        print('练习25.1-2 对所有的1<=i<=n,要求wii=0，因为结点对自身的最短路径始终为0')
        print('练习25.1-3 最短路径算法中使用的矩阵L(0)对应于常规矩阵乘法中的单位矩阵？')
        print('练习25.1-4 EXTEND-SHORTEST-PATHS所定义的矩阵乘法满足结合律')
        print('练习25.1-5 可以把单源最短路径问题表述为矩阵和向量的乘积.',
            '描述对该乘积的计算是如何与类似Bellman-Ford这样的算法相一致')
        print('练习25.1-6 希望在本节的算法中的出最短路径上的顶点。说明如何在O(n^3)时间内,',
            '根据已经完成的最短路径权值的矩阵L计算出前趋矩阵∏ 略')
        print('练习25.1-7 可以用于计算最短路径的权值相同的时间,计算出最短路径上的顶点')
        print('练习25.1-8 FASTER-ALL-PAIRS-SHORTEST-PATHS过程需要我们保存[lg(n-1)]个矩阵,',
            '每个矩阵包含n^2个元素,总的空间需求为Θ(n^2lgn),修改这个过程，',
            '使其仅使用两个n*n矩阵，需要空间为Θ(n^2)')
        print('练习25.1-9 修改FASTER-PARIS-SHORTEST-PATHS,使其能检测出图中是否存在权值为负的回路')
        print('练习25.1-10 写出一个有效的算法来计算图中最短的负权值回路的长度(即所包含的边数)')
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
        print('Floyd-Warshall算法')
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

class Chapter25_3:
    '''
    chpater25.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter25.3 note

        Example
        ====
        ```python
        Chapter25_3().note()
        ```
        '''
        print('chapter25.3 note as follow')  
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

chapter25_1 = Chapter25_1()
chapter25_2 = Chapter25_2()
chapter25_3 = Chapter25_3()

def printchapter25note():
    '''
    print chapter25 note.
    '''
    print('Run main : single chapter twenty-five!')  
    chapter25_1.note()
    chapter25_2.note()
    chapter25_3.note()

# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
if __name__ == '__main__':  
    printchapter25note()
else:
    pass
