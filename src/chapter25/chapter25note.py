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
    def solve_25_2_1(self):
        '''
        练习25.2-1
        '''
        g = _g.Graph()
        vertexs = ['1', '2', '3', '4']
        g.addvertex(vertexs)
        g.addedge('4', '1', _g.DIRECTION_TO)
        g.addedge('4', '3', _g.DIRECTION_TO)
        g.addedge('2', '4', _g.DIRECTION_TO)
        g.addedge('2', '3', _g.DIRECTION_TO)
        g.addedge('3', '2', _g.DIRECTION_TO)
        mat = g.getmatrixwithweight()
        print('带权邻接矩阵')
        print(mat)
        D_last = mat
        for i in range(g.vertex_num):
            print('第%d次迭代' % i)
            D = _esp.floyd_warshall_step(D_last, i)
            print(D)
            D_last = D

    def solve_25_2_6(self):
        '''
        练习25.2-6
        '''
        g = _g.Graph()
        vertexs = ['1', '2', '3', '4', '5']
        g.addvertex(vertexs)
        g.addedgewithdir('1', '2', 2)
        g.addedgewithdir('2', '3', 5)
        g.addedgewithdir('3', '4', 3)
        g.addedgewithdir('3', '5', -4)
        g.addedgewithdir('5', '2', -2)
        mat = g.getmatrixwithweight()
        print('带权邻接矩阵')
        print(mat)
        pi = g.getpimatrix()
        D, P = _esp.floyd_warshall(mat, pi)
        print('路径矩阵')
        print(D)
        print('前趋矩阵')
        print(P)

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
        print('采取另一种动态规划方案,解决在一个有向图G=(V,E)上每对顶点间的最短路径问题')
        print('Floyd-Warshall算法,其运行时间为Θ(V^3),允许存在权值为负的边,假设不存在权值为负的回路')
        print('最短路径结构')
        print('  在Floyd-Warshall算法中,利用最短路径结构中的另一个特征',
            '不同于基于矩阵乘法的每对顶点算法中所用到的特征')
        print('  该算法考虑最短路径上的中间顶点,其中简单路径p=<v1,v2,...,vl>',
            '上的中间顶点是除v1和vl以外p上的任何一个顶点,任何属于集合{v2,v3,...,vl-1}的顶点')
        print('  Floyd—Warshall算法主要基于以下观察.设G的顶点为V={1,2,...,n},对某个k考虑顶点的一个子集{1,2,...,k}',
            '对任意一对顶点i,j∈V，考察从i到j且中间顶点皆属于集合{1,2,...,k}的所有路径')
        print('  设p是其中一条最小权值路径(路径p是简单的),Floyd-Warshall算法利用了路径p与i到j之间的最短路径',
            '(所有中间顶点都属于集合{1,2,...,k-1})之间的联系.这一联系依赖于k是否是路径p上的一个中间顶点')
        print(' 分成两种情况：如果k不是路径p的中间顶点，则p的所有中间顶点皆在集合{1,2,...,k-1}中',
            '如果k是路径p的中间顶点,那么可将p分解为i~k~j,由引理24.1可知,p1是从i到k的一条最短路径',
            '且其所有中间顶点均属于集合{1,2,...,k-1}')
        print('解决每对顶点间最短路径问题的一个递归解')
        print('  令dij(k)为从顶点i到顶点j、且满足所有中间顶点皆属于集合{1,2,...,k}的一条最短路径的权值',
            '当k=0,从顶点i到顶点j的路径中,没有编号大于0的中间顶点')
        print('递归式')
        print('  dij(k)=wij e.g. k = 0;  dij(k)=min(dij(k-1),dik(k-1),dkj(k-1)) e.g. k >= 1;')
        print('  因为对于任意路径，所有的中间顶点都在集合{1,2,...,n}内,矩阵D(n)=dij(n)给出了最终解答',
            '对所有的i,j∈V,有dij(n)=d(i,j)')
        print('自底向上计算最短路径的权值')
        _esp.test_floyd_warshall()
        print('构造一条最短路径')
        print('  在Floyd—Warshall算法中存在大量不同的方法来建立最短路径')
        print('  一种途径是计算最短路径权值的矩阵D,然后根据矩阵D构造前趋矩阵pi。这一方法可以在O(n^3)时间内实现',
            '给定前趋矩阵pi,可以使用过程PRINT-ALL-PAIRS-SHORTEST-PATH来输出一条给定最短路径上的顶点')
        print('有向图的传递闭包')
        print('  已知一有向图G=(V,E),顶点集合V={1,2,..,n},希望确定对所有顶点对i,j∈V',
            '图G中是否都存在一条从i到j的路径.G的传递闭包定义为图G*=(V,E*)',
            '其中E*={(i,j):图G中存在一条从i到j的路径}')
        print('  在Θ(n^3)时间内计算出图的传递闭包的一种方法为对E中每条边赋以权值1,然后运行Floyd-Warshall算法',
            '如果从顶点i到顶点j存在一条路径,则dij<n。否则,有dij=∞')
        print('  另外还有一种类似的方法,可以在Θ(n^3)时间内计算出图G的传递闭包,在实际中可以节省时空需求',
            '该方法要求把Floyd-Warshall算法中的min和+算术运算操作,用相应的逻辑运算∨(逻辑OR)和∧(逻辑AND)来代替)',
            '对i,j,k=1,2,...,n,如果图G中从顶点i到顶点j存在一条通路,且其所有中间顶点均属于集合{1,2,...,k}',
            '则定义tij(k)为1,否则tij(k)为0.我们把边(i,j)加入E*中当且仅当tij(n)=1',
            '通过这种方法构造传递闭包G*=(V,E*)')
        print('tij(k)的递归定义为',
            'tij(0) = 0 e.g. 如果i!=j和(i,j)∉E',
            'tij(0) = 1 e.g. 如果i=j或(i,j)∈E',
            'k>=1, tij(k) = tij(k-1) or (tik(k-1) and tkj(k-1))')
        _esp.test_transitive_closure()
        print('练习25.2-1 代码如下')
        self.solve_25_2_1()
        print('练习25.2-2 略')
        print('练习25.2-3 证明对所有的i∈V,前趋子图Gpi,i是以i为根的一颗最短路径树')
        print('练习25.2-4 Floyd-Washall的空间复杂度可以从Θ(n^3)优化到Θ(n^2)，完成')
        print('练习25.2-5 正确')
        print('练习25.2-6 可以利用Floyd-Warshall算法的输出来检测是否存在负的回路',
            '从最终前趋矩阵看出每个路径都相同，且为负权回路')
        self.solve_25_2_6()
        print('练习25.2-7 略')
        print('练习25.2-8 写出一个运行时间为O(VE)的算法,计算有向图G=(V,E)的传递闭包')
        print('练习25.2-9 假定一个有向无环图的传递闭包可以在f(|V|,|E|)时间内计算,其中f是|V|和|E|的单调递增函数',
            '证明：计算一般有向图G=(V,E)的传递闭包G*=(V,E*)的时间为f(|V|,|E|)+O(V+E*)')
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
        print('25.3 稀疏图上的Johnson算法')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
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
