# coding:utf-8
# usr/bin/python3
# python src/chapter21/chapter21note.py
# python3 src/chapter21/chapter21note.py
'''

Class Chapter21_1

Class Chapter21_2

Class Chapter21_3

Class Chapter21_4

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

if __name__ == '__main__':
    import notintersectset as _nset
else:
    from . import notintersectset as _nset

class Chapter21_1:
    '''
    chpater21.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.1 note

        Example
        ====
        ```python
        Chapter21_1().note()
        ```
        '''
        print('chapter21.1 note as follow')  
        print('第21章 用于不相交集合的数据结构')
        print('某些应用中，要将n个不同的元素分成一组不想交的集合')
        print('不相交集合上有两个重要的操作，即找出给定的元素所属的集合和合并两个集合')
        print('为了使某种数据结构能够支持这两种操作，就需要对该数据结构进行维护；本章讨论各种维护方法')
        print('21.1描述不相交集合数据结构所支持的各种操作，并给出这种数据结构的一个简单应用')
        print('21.2介绍不想交集合的一种简单链表实现')
        print('21.3采用有根树的表示方法的运行时间在实践上来说是线性的，但从理论上来说是超线性的')
        print('21.4定义并讨论一种增长极快的函数以及增长极为缓慢的逆函数')
        print(' 在基于树的实现中，各操作的运行时间中都出现了该反函数。',
            '然后，再利用平摊分析方法，证明运行时间的一个上界是超线性的')
        print('21.1 不相交集合上的操作')
        print('不相交集合数据结构保持一组不相交的动态集合S={S1,S2,...,Sk}')
        print('每个集合通过一个代表来识别，代表即集合中的某个成员')
        print('集合中的每一个元素是由一个对象表示的，设x表示一个对象，希望支持以下操作')
        print('MAKE-SET(x)：其唯一成员(因而其代表)就是x。因为各集合是不相交的，故要求x没有在其他集合出现过')
        print('UNION(x, y)：将包含x和y的动态集合(比如说Sx和Sy)合并为一个新的集合(并集)')
        print('FIND-SET(x)：返回一个指针，指向包含x的(唯一)集合的代表')
        print('不相交集合数据结构的一个应用')
        # !不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数
        print(' 不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数')
        g = _nset.UndirectedGraph()
        g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        g.edges = [('d', 'i'), ('f', 'k'), ('g', 'i'),
                   ('b', 'g'), ('a', 'h'), ('i', 'j'), 
                   ('d', 'k'), ('b', 'j'), ('d', 'f'), 
                   ('g', 'j'), ('a', 'e'), ('i', 'd')]
        print('练习21.1-1') 
        print(' 无向图G=(V, E)的顶点集合V=')
        print('  {}'.format(g.vertexs))
        print(' 边集合E=')
        print('  {}'.format(g.edges))
        print(' 所有连通子图的顶点集合为')
        print('  {}'.format(g.get_connected_components()))
        print('练习21.1-2 证明：在CONNECTED-COMPONENTS处理了所有的边后')
        print(' 两个顶点在同一个连通子图中，当且仅当它们在同一个集合中')
        print('练习21.1-3 无向图G=(V,E)调用FIND_SET的次数为len(E)*2,',
            '调用UNION的次数为len(V) - k')
        g.print_last_connected_count()
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_2:
    '''
    chpater21.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.2 note

        Example
        ====
        ```python
        Chapter21_2().note()
        ```
        '''
        print('chapter21.2 note as follow')
        print('21.2 不相交集合的链表表示')
        print(' 要实现不相交集合数据结构，一种简单的方法是每一种集合都用一个链表来表示')
        print(' 每个链表中的第一个对象作为它所在集合的代表')
        print(' 链表中的每一个对象都包含一个集合成员，一个指向包含下一个集合成员的对象的指针，以及指向代表的指针')
        print(' 每个链表都含head指针和tail指针，head指向链表的代表，tail指向链表中最后的对象')
        print(' 在这种链表表示中，MAKE-SET操作和FIND-SET操作都比较容易实现，只需要O(1)的时间')
        print(' 执行MAKE-SET(x)操作，创建一个新的链表,其仅有对象为x')
        print(' 对FIND-SET(X)操作,只要返回由x指向代表的指针即可')
        print('合并的一个简单实现')
        print(' 在UNION操作的实现中，最简单的是采用链表集合表示的实现，',
            '这种实现要比MAKE-SET或FIND-SET多不少的时间')
        print(' 执行UNION(x,y),就是将x所在的链表拼接到y所在链表的表尾.利用y所在链表的tail指针',
            '可以迅速地找到应该在何处拼接x所在的链表')
        print(' 一个作用于n个对象上的,包含m个操作的序列，需要Θ(n^2)时间')
        print(' 执行n个MAKE-SET操作所需要的时间为Θ(n)')
        print(' 因为第i个UNION操作更新了i个对象，故n-1个UNION操作所更新的对象总数为Θ(n^2)')
        print(' 总的操作数为2n-1，平均来看，每个操作需要Θ(n)的时间')
        print(' 也就是一个操作的平摊时间为Θ(n)')
        print('一种加权合并启发式策略')
        print(' 在最坏情况下，根据上面给出的UNION过程的实现，每次调用这一过程都需要Θ(n)的时间')
        print(' 如果两个表一样长的话，可以以任意顺序拼接，利用这种简单的加权合并启发式策略')
        print(' 如果两个集合都有Ω(n)个成员的话，一次UNION操作仍然会需要Θ(n)时间')
        print('定理21.1 利用不相交集合的链表表示和加权合并启发式',
            '一个包括m个MAKE-SET,UNION和FIND-SET操作',
            '(其中有n个MAKE-SET操作)的序列所需时间为O(m+nlgn)')
        print('练习21.2-1 已经完成')
        print('练习21.2-2 如下:结果是16个链表集合合并成了一个总的链表')
        _nset.test_list_set()
        print('练习21.2-3 对定理21.1的证明加以改造，使得MAKE-SET和FIND-SET操作有平坦时间界O(1)')
        print(' 对于采用了链表表示和加权合并启发式策略的UNION操作，有界O(lgn)')
        print('练习21.2-4 假定采用的是链表表示和加权合并启发式策略。略')
        print('练习21.2-5 合并的两个表像合并排序那样轮流交叉合并')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_3:
    '''
    chpater21.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.3 note

        Example
        ====
        ```python
        Chapter21_3().note()
        ```
        '''
        print('chapter21.3 note as follow')
        print('21.3 不相交集合森林')
        print('在不相交集合的另一种更快的实现中，用有根树来表示集合，',
            '树中的每个结点都包含集合的一个成员，每棵树表示一个集合')
        print('在不相交集合森林中，每个成员仅指向其父结点。',
            '每棵树的根包含了代表，并且是它自己的父亲结点')
        print('尽管采用了这种表示的直观算法并不比采用链表表示的算法更快')
        print('通过引入两种启发式策略(\"按秩合并\"和\"路径压缩\")',
            '就可以获得目前已知的、渐进意义上最快的不相交集合数据结构了')
        print('不相交集合森林。MAKE-SET创建一棵仅包含一个结点的树')
        print('在执行FIND-SET操作时，要沿着父结点指针一直找下去，直到找到树根为止')
        print('在这一查找路径上访问过的所有结点构成查找路径(find path),UNION操作使得一棵树的根指向另一颗树的根')
        print('改进运行时间的启发式策略')
        print(' 还没有对链表实现做出改进。一个包含n-1次UNION操作的序列可能会构造出一棵为n个结点的线性链的树')
        print(' 通过采用两种启发式策略，可以获得一个几乎与总的操作数m成线性关系的运行时间')
        print(' 1.按秩合并(union by rank),与我们用于链表表示中的加权合并启发式是相似的')
        print('   其思想是使包含较少结点的树的根指向包含较多结点的树的根')
        print('   对每个结点，用秩表示结点高度的一个上界。在按秩合并中，具有较小秩的根的UNION操作中要指向具有较大秩的根')
        print(' 2.路径压缩(path compression),非常简单有效,在FIND-SET操作中,',
            '利用这种启发式策略,来使查找路径上的每个结点都直接指向根结点,路径压缩并不改变结点的秩')
        print('启发式策略对运行时间的影响')
        print(' 如果将按秩合并或路径压缩分开来使用的话，都能改善不相交集合森林操作的运行时间')
        print(' 如果将这两种启发式合起来使用，则改善的幅度更大。')
        print(' 单独来看,按秩合并产生的运行时间为O(mlgn),这个界是紧确的')
        print(' 如果有n个MAKE-SET操作和f个FIND-SET操作,则单独应用路径压缩启发式的话')
        print(' 得到的最坏情况运行时间为Θ(n+f*(1+log2+f/n(n)))')
        print('当同时使用按秩合并和路径压缩时，最坏情况运行时间为O(ma(n)),',
            'a(n)是一个增长及其缓慢的函数')
        print(' 在任意可想象的不相交集合数据结构的应用中，都会有a(n)<=4',
            '在各种实际情况中，可以把这个运行时间看作与m成线性关系')
        print('练习21.3-1 用按秩合并和路径压缩启发式的不相交集合森林重做练习21.2-2')
        _nset.test_forest_set()
        print('练习21.3-2 写出FIND-SET的路径压缩的非递归版本')
        print('练习21.3-3 请给出一个包含m个MAKE-SET，UNION和FIND-SET操作的序列',
              '(其中n个是MAKE-SET操作)，使得采用按秩合并时，这一操作序列的时间代价为Ω(mlgn)')
        print('练习21.3-4 证明：在采用了按秩合并和路径压缩时，',
            '任意一个包含m个MAKE-SET,FIND-SET和LINK操作的序列',
            '(其中所有LINK操作出现于FIND-SET操作之前)需要O(m)的时间')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_4:
    '''
    chpater21.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.4 note

        Example
        ====
        ```python
        Chapter21_4().note()
        ```
        '''
        print('chapter21.4 note as follow')
        print('21.4 带路径压缩的按秩合并的分析')
        # !对作用于n个元素上的m个不相交集合操作，
        # !联合使用按秩合并和路径压缩启发式的运行时间为O(ma(n))
        print('对作用于n个元素上的m个不相交集合操作,',
            '')
        print('联合使用按秩合并和路径压缩启发式的运行时间为O(ma(n))')
        print('秩的性质')
        print('引理21.4 对所有的结点x，有rank[x]<=rank[p[x]],如果x!=p[x]则不等号严格成立')
        print(' rank[x]的初始值位0，并随时间而增长，直到x!=p[x];从此以后rank[x]就不再变化')
        print(' rank[p[x]]的值是时间的单调递增函数')
        print('推论21.5 在从任何一个结点指向根的路径上，结点的秩是严格递增的')
        print('引理21.6 每个结点的秩至多为1')
        print('时间界的证明')
        print('引理21.7 假定有一个m个MAKE-SET,UNION和FIND-SET')
        print('')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

chapter21_1 = Chapter21_1()
chapter21_2 = Chapter21_2()
chapter21_3 = Chapter21_3()
chapter21_4 = Chapter21_4()

def printchapter21note():
    '''
    print chapter21 note.
    '''
    print('Run main : single chapter twenty-one!')  
    chapter21_1.note()
    chapter21_2.note()
    chapter21_3.note()
    chapter21_4.note()

# python src/chapter21/chapter21note.py
# python3 src/chapter21/chapter21note.py
if __name__ == '__main__':  
    printchapter21note()
else:
    pass
