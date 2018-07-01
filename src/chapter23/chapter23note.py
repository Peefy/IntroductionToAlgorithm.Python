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
    import mst as _mst
else: 
    from . import mst as _mst

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
        print('设计电子线路时，如果要使n个引脚互相连通,可以使用n-1条连接线',
            '每条连接线连接两个引脚。在各种链接方案中，通常希望找出连接线最少的接法')
        print('可以把这一接线问题模型化为一个无向连通图G=(V,E)',
            '其中V是引脚集合，E是每对引脚之间可能互联的集合')
        print('对图中每一条边(u,v)∈E,都有一个权值w(u,v)表示连接u和v的代价(需要的接线数目)')
        print('希望找出一个无回路的子集T∈E,它连接了所有的顶点，且其权值之和w(T)=∑w(u,v)最小')
        print('因为T无回路且连接了所有的顶点,所以它必然是一棵树，称为生成树')
        print('因为由最小生成树可以\"生成\"图G')
        print('把确定树T的问题称为最小生成树问题')
        print('最小生成树问题的两种算法：Kruskal算法和Prim算法')
        print('这两种算法中都使用普通的二叉堆，都很容易达到O(ElgV)的运行时间')
        print('通过采用斐波那契堆，Prim算法的运行时间可以减小到O(E+VlgV),',
            '如果|V|远小于|E|的话,这将是对该算法的较大改进')
        print('这两个算法都是贪心算法，在算法的每一步中，都必须在几种可能性中选择一种')
        print('贪心策略的思想是选择当时最佳的可能性，一般来说，这种策略不一定能保证找到全局最优解')
        print('然而，最小生成树问题来说,却可以证明某些贪心策略的确可以获得具有最小权值的生成树')
        print('23.1 最小生成树的形成')
        print('假设已知一个无向连通图G=(V,E),其权值函数为w')
        print('目的是找到图G的一棵最小生成树')
        print('通用最小生成树算法')
        print('在每一个步骤中都形成最小生成树的一条边,算法维护一个边的集合A,保持以下的循环不变式:')
        print(' 在每一次循环迭代之前，A是某个最小生成树的一个子集')
        print(' 在算法的每一步中，确定一条边(u,v)，使得将它加入集合A后，仍然不违反之歌循环不变式;',
            '亦即，A∪{(u,v)}仍然是某一个最小生成树的子集')
        print(' 称这样的边为A的安全边(safe edge),因为可以安全地把它添加到A中,而不会破坏上述的循环不变式')
        print('在算法的执行过程中，集合A始终是无回路的，否则包含A的最小生成树将包含一个环')
        print('无向图G=(V, E)的一个割(S, V-S)是对V的一个划分.当一条边(u,v)∈E的一个端点属于S，而另一个端点属于V-S',
            '则称边(u,v)通过割(S,V-S).如果一个边的集合A中没有边通过某一割','则说该割不妨害边集A')
        print('如果某条边的权值是通过一个割的所有边中最小的,则称该边为通过这个的割的一条轻边(light edge)')
        print('GENERIC-MST')
        print('  A = []')
        print('  while A does not form a spanning tree')
        print('    do find an edge (u,v) that is safe for A (保证不形成回路)')
        print('       A <- A ∪ {(u, v)}')
        print('  return A')
        print('')
        print('识别安全边的一条规则：')
        print('定理23.1 设图G=(V,E)是一个无向连通图，并且在E上定义了一个具有实数值的加权函数w.',
            '设A是E的一个子集，它包含于G的某个最小生成树中.',
            '设割(S,V-S)是G的任意一个不妨害A的割,且边(u,v)是通过集合A来说是安全的')
        print('推论23.2 设G=(V,E)是一个无向连通图,并且在E上定义了相应了实数值加权函数w',
            '设A是E的子集，且包含于G的某一最小生成树中。设C=(Vc,Ec)为森林GA=(V,A)的一个连通分支(树)',
            '如果边(u,v)是连接C和GA中其他某联通分支的一条轻边,则(u,v)对集合A来说是安全的')
        print('证明:因为割(Vc,V-Vc)不妨害A，(u,v)是该割的一条轻边。因此(u,v)对A来说是安全的')
        print('练习23.1-1 设(u,v)是图G中的最小权边.证明:(u,v)属于G的某一棵最小生成树')
        print('练习23.1-2 略')
        print('练习23.1-3 证明：如果一条边(u,v)被包含在某一最小生成树中,那么它就是通过图的某个割的轻边')
        print('练习23.1-4 因为这条边虽然是轻边，但是连接后产生不安全的回路')
        print('练习23.1-5 设e是图G=(V,E)的某个回路上一条最大权边.证明：存在着G\'=(V,E-{e})的一棵最小生成树,',
            '它也是G的最小生成树。亦即，存在着G的不包含e的最小生成树')
        print('练习23.1-6 证明：一个图有唯一的最小生成树,如果对于该图的每一个割,都存在着通过该割的唯一一条轻边',
            '但是其逆命题不成立')
        print('练习23.1-7 论证：如果图中所有边的权值都是正的，那么，任何连接所有顶点、',
            '且有着最小总权值的边的子集必为一棵树')
        print('练习23.1-8 设T是图G的一棵最小生成树，L是T中各边权值的一个已排序的列表',
            '证明：对于G的任何其他最小生成树T\'，L也是T\'中各边权值的一个已排序的列表')
        print('练习23.1-9 设T是图G=(V,E)的一棵最小生成树,V\'是V的一个子集。设T\'为T的一个基于V\'的子图',
            'G\'为G的一个基于V\'的子图。证明:如果T\'是连通的,则T\'是G\'的一棵最小生成树')
        print('练习23.1-10 给定一个图G和一棵最小生成树T,假定减小了T中某一边的权值。',
            '证明：T仍然是G的一棵最小生成树。更形式地,设T是G的一棵最小生成树',
            '其各边的权值由权值函数w给出.')
        print(' 证明：T是G的一棵最小生成树，其各边的权值由w\'给出')
        print('练习23.1-11 给定一个图G和一棵最小生成树T，假定减小了不在T中的某条边的权值',
            '请给出一个算法,来寻找经过修改的图中的最小生成树')
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
        print('23.2 Kruskai算法和Prim算法')
        print('本节所介绍的两种最小生成树算法是对上一节介绍的通用算法的细化')
        print('均采用了一个特定的规则来确定GENERIC-MST算法所描述的安全边')
        print(' 在Kruskal算法中,集合A是一个森林,加入集合A中的安全边总是图中连接两个不同连通分支的最小权边')
        print(' 在Prim算法中,集合A仅形成单棵树,',
            '添加入集合A的安全边总是连接树与一个不在树中的顶点的最小权边')
        print('Kruskal算法')
        print(' 该算法找出森林中连接任意两棵树的所有边中,具有最小权值的边(u,v)作为安全边',
            '并把它添加到正在生长的森林中')
        print(' 设C1和C2表示边(u,v)连接的两棵树,因为(u,v)必是连接C1和其他某棵树的一条轻边',
            '所以由推论23.2可知,(u,v)对C1来说是安全边。Kruskal算法同时也是一种贪心算法',
            '因为在算法的每一步中,添加到森林中的边的权值都是尽可能小的')
        _mst.test_mst_kruskal()
        print(' Kruskal算法在图G=(V,E)上的运行时间取决于不相交集合数据结构是如何实现的')
        print(' 使用*按秩结合*和*路径压缩*的启发式方法实现不相交集合森林，从渐进意义上来说是最快的方法')
        print('')
        print('')
        print('')
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
