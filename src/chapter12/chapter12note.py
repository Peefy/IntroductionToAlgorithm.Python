
# python src/chapter12/chapter12note.py
# python3 src/chapter12/chapter12note.py
'''
Class Chapter12_1

Class Chapter12_2

Class Chapter12_3

Class Chapter12_4

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter12_1:
    '''
    chpater12.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.1 note

        Example
        ====
        ```python
        Chapter12_1().note()
        ```
        '''
        print('chapter12.1 note as follow')
        print('查找树(search tree)是一种数据结构，它支持多种动态集合操作，', 
            '包括SEARCH,MINIMUM,MAXIMUM,PREDECESSOR,SUCCESSOR,INSERT以及DELETE,', 
            '它既可以用作字典，也可以用作优先队列')
        print('在二叉查找树(binary search tree)上执行的基本操作时间与树的高度成正比')
        print('对于一颗含n个结点的完全二叉树，这些操作的最坏情况运行时间为Θ(lgn)')
        print('但是，如果树是含n个结点的线性链，则这些操作的最坏情况运行时间为Θ(n)')
        print('在12.4节中可以看到，一棵随机构造的二叉查找树的期望高度为O(lgn)，', 
            '从而这种树上基本动态集合操作的平均时间为Θ(lgn)')
        print('在实际中，并不总能保证二叉查找树是随机构造成的，但对于有些二叉查找树的变形来说，')
        print(' 各基本操作的最坏情况性能却能保证是很好的')
        print('第13章中给出这样一种变形，即红黑树，其高度为O(lgn)。第18章介绍B树，这种结构对维护随机访问的二级(磁盘)存储器上的数据库特别有效')
        print('12.1 二叉查找树')
        print('一颗二叉查找树是按二叉树结构来组织的。这样的树可以用链表结构表示，其中每一个结点都是一个对象。')
        print('结点中除了key域和卫星数据外，还包含域left,right和p，它们分别指向结点的左儿子、右儿子和父节点。')
        print('如果某个儿子结点或父节点不存在，则相应域中的值即为NIL，根结点是树中唯一的父结点域为NIL的结点')
        print('二叉查找树，对任何结点x，其左子树中的关键字最大不超过key[x],其右子树中的关键字最小不小于key[x]')
        print('不同的二叉查找树可以表示同一组值，在有关查找树的操作中，大部分操作的最坏情况运行时间与树的高度是成正比的')
        print('二叉查找树中关键字的存储方式总是满足以下的二叉树查找树性质')
        print('')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_2:
    '''
    chpater12.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.2 note

        Example
        ====
        ```python
        Chapter12_2().note()
        ```
        '''
        print('chapter12.2 note as follow')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_3:
    '''
    chpater12.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.3 note

        Example
        ====
        ```python
        Chapter12_3().note()
        ```
        '''
        print('chapter12.3 note as follow')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_4:
    '''
    chpater12.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.4 note

        Example
        ====
        ```python
        Chapter12_4().note()
        ```
        '''
        print('chapter12.4 note as follow')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

chapter12_1 = Chapter12_1()
chapter12_2 = Chapter12_2()
chapter12_3 = Chapter12_3()
chapter12_4 = Chapter12_4()

def printchapter12note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter twelve!')  
    chapter12_1.note()
    chapter12_2.note()
    chapter12_3.note()
    chapter12_4.note()

# python src/chapter12/chapter12note.py
# python3 src/chapter12/chapter12note.py
if __name__ == '__main__':  
    printchapter12note()
else:
    pass
