# coding:utf-8
# usr/bin/python3
# python src/chapter18/chapter18note.py
# python3 src/chapter18/chapter18note.py
'''

Class Chapter18_1

Class Chapter18_2

Class Chapter18_3

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
    import btree as bt
else:
    from . import btree as bt

class Chapter18_1:
    '''
    chpater18.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.1 note

        Example
        ====
        ```python
        Chapter18_1().note()
        ```
        '''
        print('chapter18.1 note as follow')  
        print('第五部分 高级数据结构')
        # !B树是一种被设计成专门存储在磁盘上的平衡查找树
        print('第18章 B树是一种被设计成专门存储在磁盘上的平衡查找树')
        print(' 因为磁盘的操作速度要大大慢于随机存取存储器，所以在分析B树的性能时')
        print(' 不仅要看动态集合操作花了多少计算时间，还要看执行了多少次磁盘存取操作')
        print(' 对每一种B树操作，磁盘存取的次数随B树高度的增加而增加，而各种B树操作又能使B树保持较低的高度')
        print('第19章,第20章 给出可合并堆的几种实现。')
        print(' 这种堆支持操作INSERT,MINIMUM,EXTRACT-MIN和UNION.')
        print(' UNION操作用于合并两个堆。这两章中出现的数据结构还支持DELETE和DECREASE-KEY操作')
        print('第19章中出现的二项堆结构能在O(lgn)最坏情况时间内支持以上各种操作，此处n位输入堆中的总元素数')
        print('第20章 斐波那契堆对二项堆进行了改进 操作INSERT,MINIMUM和UNION仅花O(1)的实际和平摊时间')
        print(' 操作EXTRACT-MIN和DELETE要花O(lgn)的平摊时间')
        # !渐进最快的图问题算法中，斐波那契堆是其核心部分
        print(' 操作DECREASE-KEY仅花O(1)的平摊时间')
        print('第21章 用于不想交集合的一些数据结构，由n个元素构成的全域被划分成若干动态集合')
        print(' 一个由m个操作构成的序列的运行时间为O(ma(n)),其中a(n)是一个增长的极慢的函数')
        print(' 在任何可想象的应用中，a(n)至多为4.')
        print(' 这个问题的数据结构简单，但用来证明这个时间界的平摊分析却比较复杂')
        print('其他一些高级的数据结构：')
        print(' 动态树：维护一个不相交的有根树的森林')
        print('  在动态树的一种实现中，每个操作具有O(lgn)的平摊时间界；',
            '在另一种更复杂的实现中，最坏情况时间界O(lgn).动态树常用在一些渐进最快的网络流算法中')
        print(' 伸展树：是一种二叉查找树，标准的查找树操作在其上以O(lgn)的平摊时间运行,',
            '伸展树的一个应用是简化动态树')
        print(' 持久的数据结构允许在过去版本的数据结构上做查询，甚至有时候做更新,',
            '只需很小的时空代价，就可以使链式数据结构持久化的技术')
        print('第18章 B 树')
        # !B树是为磁盘或其他直接存取辅助设备而设计的一种平衡查找树
        print('B树是为磁盘或其他直接存取辅助设备而设计的一种平衡查找树。与红黑树类似，',
            '但是在降低磁盘I/O操作次数方面更好一些。许多数据库系统使用B树或者B树的变形来存储信息')
        # !B树与红黑树的主要不同在于，B树的结点可以有许多子女，从几个到几千个，就是说B树的分支因子可能很大
        print('B树与红黑树的主要不同在于，B树的结点可以有许多子女，从几个到几千个，就是说B树的分支因子可能很大')
        print('这一因子常常是由所使用的磁盘特性所决定的。')
        print('B树与红黑树的相似之处在于，每棵含有n个结点的B树高度为O(lgn),',
            '但可能要比一棵红黑树的高度小许多，因为分支因子较大')
        print('所以B树也可以被用来在O(lgn)时间内，实现许多动态集合操作')
        print('B树以自然的方式推广二叉查找树。如果B树的内结点x包含x.n个关键字，则x就有x.n+1个子女')
        print('结点x中的关键字是用来将x所处理的关键字域划分成x.n+1个子域的分隔点，每个子域都由x中的一个子女来处理')
        print('铺存上的数据结构')
        print(' 有许多不同的技术可用来在计算机中提供存储能力')
        print(' 典型的磁盘驱动器，这个驱动器包含若干盘片，它们以固定速度绕共用的主轴旋转')
        print('虽然磁盘比主存便宜而且有搞笑的容量，但是它们速度很慢')
        print('有两种机械移动的成分：盘旋转和磁臂移动')
        print('在一个典型的B树应用中，要处理的数据量很大，因此无法一次都装入主存')
        print('B树算法将所需的页选择出来复制到主存中去，而后将修改过的页再写回到磁盘上去')
        print('18.1 B树的定义')
        print('一棵B树T是具有如下性质的有根树(根为root[T]):')
        print('1) 每个结点x有以下域')
        print(' a) n[x],当前存储在结点x中的关键字数')
        print(' b) n[x]个关键字本身，以非降序存放，因此key1[x]<=key2[x]<=...<=keyn[x]')
        print(' c) leaf[x],是一个布尔值，如果x是叶子的话，则它为TRUE,如果x为一个内结点，则为FALSE')
        print('2) 每个内结点x还包含n[x]+1个指向其子女的指针c1[x],c2[x],...,cn[x]+1[x].叶结点没有子女，故它们的ci域无定义')
        print('3) 各关键字keyi[x]对存储在各子树中的关键字范围加以分隔：如果ki为存储在以ci[x]为根的子树中的关键字')
        print('4) 每个叶结点具有相同的深度，即树的高度h')
        print('5) 每一个节点能包含的关键字数有一个上界和下界。这些界可用一个称作B树的最小度数的固定整数t>=2来表示')
        print(' a) 每个非根的结点必须至少有t-1个关键字。每个非根的内结点至少有t个子女。',
            '如果树是非空的，则根节点至少包含一个关键字')
        print(' b) 每个结点可包含至多2t-1个关键字。所以一个内结点至多可有2t个子女。')
        print('   如果某结点恰好2t-1个关键字，则根节点至少包含一个关键字')
        print('t=2时的B树(二叉树)是最简单的。这时每个内结点有2个、3个或者4个子女，亦即一棵2-3-4树。',
            '然而在实际中，通常是采用大得多的t值')
        print('B树的高度')
        print(' B树上大部分操作所需的磁盘存取次数与B树的高度成正比')
        print('定理18.1 如果n>=1,则对任意一棵包含n个关键字、高度为h、最小度数t>=2的B树T，有：h<=logt((n + 1) / 2)')
        print('证明：如果一棵B树的高度为h,其根结点包含至少一个关键字而其他结点包含至少t-1个关键字。',
            '这样，在深度1至少有两个结点，在深度2至少有2t个结点，',
                '在深度3至少有2t^2个结点，直到深度h至少有2t^(h-1)个结点。')
        print('与红黑树和B树相比，虽然两者的高度都以O(lgn)的速度增长，对B树来说对数的底要大很多倍')
        print('对大多数的树操作来说，要查找的结点数在B树中要比在红黑树中少大约lgt个因子')
        print('因为在树中查找任意一个结点通常都需要一次磁盘存取，所以磁盘存取的次数大大地减少了')
        print('练习18.1-1 为什么不允许B树的最小度数t为1，t最小只能取2(二叉树)，再小就构不成树了')
        print('练习18.1-2 t的取值为t>=2,才能使图中的树是棵合法的B树')
        print('练习18.1-3 二叉树，二叉查找树，2-3-4树都可以成为最小度数为2的所有合法B树')
        print('练习18.1-4 由定理18.1中的公式h<=logt((n + 1) / 2)得，',
            '一棵高度为h的B树中，可以最多存储[2t^h-1]个结点')
        print('练习18.1-5 如果红黑树中的每个黑结点吸收它的红子女，并把它们的子女并入自身，描述这个结果的数据结构')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_2:
    '''
    chpater18.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.2 note

        Example
        ====
        ```python
        Chapter18_2().note()
        ```
        '''
        print('chapter18.2 note as follow')
        print('18.2 对B树的基本操作')
        print('这一节给出操作B-TREE-SEARCH, B-TREE-CREATE和B-TREE-INSERT的细节，两个约定:')
        print(' (1) B树的根结点始终在主存中，因而无需对根做DISK-DEAD；但是，',
            '每当根结点被改变后，都需要对根结点做一次DISK-WRITE')
        print(' (2) 任何被当做参数的结点被传递之前，要先对它们做一次DISK-READ')
        print('给出的过程都是\"单向\"算法，即它们沿树的根下降，没有任何回溯')
        print('搜索B树')
        print(' 搜索B树有搜索二叉查找树很相似，只是不同于二叉查找树的两路分支，而是多路分支')
        print(' 即在每个内结点x处，要做x.n+1路的分支决定')
        print(' B-TREE-SEARCH是定义在二叉查找树上的TREE-SEARCH过程的一个直接推广。',
            '它的输入是一个指向某子树的根结点x的指针，以及要在该子树中搜索的一个关键字k',
            '顶层调用的形式为B-TREE-SEARCH(root, key).如果k在B树中，',
            'B-TREE-SEARCH就返回一个由结点y和使keyi[y]==k成立的下标i组成的有序对(y, i)',
            '否则返回NONE')
        print(' 像在二叉查找树的TREE-SEARCH过程中那样，在递归过程中所遇到的结点构成以一条从树根下降的路径')
        print('创建一棵空的B树')
        print(' 为构造一棵B树T，先用B-TREE-CREATE来创建一个空的根结点，再调用B-TREE-INSERT来加入新的关键字')
        print('向B树插入关键字')
        print(' 与向二叉查找树中插入一个关键字相比向B树中插入一个关键字复杂得多。')
        print(' 像在二叉查找树中一样，要查找插入新关键字的叶子位置。',
            '但是在B树中，不能简单地创建一个新的叶结点，然后将其插入,因为这样得到的树不再是一颗有效的B树')
        print('B树中结点的分裂')
        print('')
        print('')
        # btree = bt.BTree.create()
        # print(btree.root)
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_3:
    '''
    chpater18.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.3 note

        Example
        ====
        ```python
        Chapter18_3().note()
        ```
        '''
        print('chapter18.3 note as follow')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

chapter18_1 = Chapter18_1()
chapter18_2 = Chapter18_2()
chapter18_3 = Chapter18_3()

def printchapter18note():
    '''
    print chapter18 note.
    '''
    print('Run main : single chapter eighteen!')  
    chapter18_1.note()
    chapter18_2.note()
    chapter18_3.note()

# python src/chapter18/chapter18note.py
# python3 src/chapter18/chapter18note.py
if __name__ == '__main__':  
    printchapter18note()
else:
    pass
