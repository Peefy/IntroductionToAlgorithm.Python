# coding:utf-8
# usr/bin/python3
# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
'''

Class Chapter19_1

Class Chapter19_2

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

class Chapter19_1:
    '''
    chpater19.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter19.1 note

        Example
        ====
        ```python
        Chapter19_1().note()
        ```
        '''
        print('chapter19.1 note as follow')  
        print('第19章 二项堆')
        print('可合并堆(包括二叉堆、二项堆、斐波那契堆)的数据结构，这些数据结构支持下面五种操作')
        print('MAKE-HEAP():创建并返回一个不包含任何元素的新堆')
        print('INSERT(H,x):将结点x(其关键字域中已填入了内容)插入堆H中')
        print('MINIMUM(H):返回一个指向堆H中包含最小关键字的结点的指针')
        print('EXTRACT-MIN(H):将堆H中包含的最小关键字删除，并返回一个指向该结点的指针')
        print('UNION(H1,H2):创建并返回一个包含堆H1和H2中所有结点的新堆。同时H1和H2被这个操作\"删除\"')
        print('DECREASE-KEY(H, x, k):将新关键字值k(假定它不大于当前的关键字值)赋给堆H中的结点x')
        print('DELETE(H, x):从堆H中删除结点x')
        print('    过程       |二叉堆(最坏情况)|二项堆(最坏情况)|斐波那契堆(平摊)|')
        print(' MAKE-HEAP()   |     Θ(1)      |      Θ(1)    |     Θ(1)      |')
        print(' INSERT(H,x)   |    Θ(lgn)     |     Ω(lgn)   |     Θ(1)      |')
        print(' MINIMUM(H)    |     Θ(1)      |     Ω(lgn)   |     Θ(1)      |')
        print(' EXTRACT-MIN(H)|    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print(' UNION(H1,H2)  |     Θ(n)      |     Ω(lgn)   |     Θ(1)      |')
        print(' DECREASE-KEY  |    Θ(lgn)     |     Θ(lgn)   |     Θ(1)      |')
        print(' DELETE(H, x)  |    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print('对操作SEARCH操作的支持方面看，二叉堆、二项堆、斐波那契堆都是低效的')
        print('19.1 二项树和二项堆')
        print('19.1.1 二项树')
        # !二项树Bk是一种递归定义的树。
        print('二项树Bk是一种递归定义的树。')
        # !二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子
        print('二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子')
        print('引理19.1(二项树的性质) 二项树Bk具有以下的性质')
        print('1) 共有2^k个结点')
        print('2) 树的高度为k')
        print('3) 在深度i处恰有(k i)个结点，其中i=0,1,2,...,k')
        print('4) 根的度数为k，它大于任何其他结点的度数；',
            '并且，如果根的子女从左到右编号为k-1,k-2,...,0,子女i是子树Bi的根')
        print('推论19.2 在一棵包含n个结点的二项树中，任意结点的最大度数为lgn')
        print('19.1.2 二项堆')
        print('二项堆H由一组满足下面的二项堆性质的二项树组成')
        print('(1) H中的每个二项树遵循最小堆性质：',
            '结点的关键字大于或等于其父结点的关键字,我们说这种树是最小堆有序的')
        print('(2) 对任意非负整数k，在H中至多有一棵二项树的根具有度数k')
        print('在一棵最小堆有序的二项树中，其根包含了树中最小的关键字')
        print('在包含n个结点的二项堆H中，包含至多[lgn]+1棵二项树')
        print('这样，二项堆H包含至多[lgn]+1棵二项树')
        print('包含13个结点的二项堆H。13的二进制表示为1101，',
            '故H包含了最小堆有序二项树B3,B2和B0,它们分别有8,4,1个结点，即共有13个结点')
        print('二项堆的表示')
        print(' 在二项堆的每个结点中，都有一个关键字域及其其他依应用要求而定的卫星数据')
        print(' 另外，每个结点x还包含了指向其父结点的指针p[x],指向其最做孩子的指针child[x]')
        print(' 以及指向x的紧右兄弟的指针sibling[x].如果结点x是根，则p[x]=None')
        print(' 如果结点x没有子女，则child[x]=None,如果x是其父结点的最右孩子，则sibling[x]=None')
        print(' 如果结点x是根，则p[x]=None,如果结点x没有子女，',
            '则child[x]=None,如果x是其父结点的最右孩子，',
            '则sibling[x]=None,每个结点x都包含域degree[x],即x的子女个数')
        print('一个二项堆中的各二项树被组织成一个链表，我们称之为根表。')
        print('在遍历根表时，各根的度数是严格递增的')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

class Chapter19_2:
    '''
    chpater19.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter19.2 note

        Example
        ====
        ```python
        Chapter19_2().note()
        ```
        '''
        print('chapter19.2 note as follow')  
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

chapter19_1 = Chapter19_1()
chapter19_2 = Chapter19_2()

def printchapter19note():
    '''
    print chapter19 note.
    '''
    print('Run main : single chapter nineteen!')  
    chapter19_1.note()
    chapter19_2.note()

# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
if __name__ == '__main__':  
    printchapter19note()
else:
    pass
