
# python src/chapter12/chapter12note.py
# python3 src/chapter12/chapter12note.py
'''
Class Chapter13_1

Class Chapter13_2

Class Chapter13_3

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter13_1:
    '''
    chpater13.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.1 note

        Example
        ====
        ```python
        Chapter13_1().note()
        ```
        '''
        print('chapter13.1 note as follow')
        print('第13章 红黑树')
        print('由12章可以知道，一棵高度为h的查找树可以实现任何一种基本的动态集合操作')
        print('如SEARCH,PREDECESOR,MINIMUM,MAXIMUM,DELETE,INSERT,其时间都是O(h)')
        print('所以，当树的高度比较低时，以上操作执行的就特别快')
        print('当树的高度比较高时，二叉查找树的性能可能不如链表好，')
        print('红黑树是许多平衡的查找树中的一种，能保证在最坏情况下，基本的动态集合操作的时间为O(lgn)')
        print('13.1 红黑树的性质')
        print('红黑树是一种二叉查找树，但在每个结点上增加一个存储位表示结点的颜色，可以是RED，可以是BLACK')
        print('通过对任何一条从根到叶子的路径上各个结点的着色方式的限制，红黑树确保没有一条路径会比其他路径长出两倍，因而是接近平衡的')
        print('树中每个结点包含5个域，color,key,p,left,right')
        print('如果某结点没有子结点或者父结点，则该结点相应的域为NIL')
        print('将NONE看作指向二叉查找树的外结点，把带关键字的结点看作树的内结点')
        print('一颗二叉查找树满足下列性质，则为一颗红黑树')
        print('1.每个结点是红的或者黑的')
        print('2.根结点是黑的')
        print('3.每个外部叶结点(NIL)是黑的')
        print('4.如果一个结点是红的，则它的两个孩子都是黑的')
        print('5.对每个结点，从该结点到其子孙结点的所有路径上包含相同数目的黑结点')
        print('为了处理红黑树代码中的边界条件，采用一个哨兵来代替NIL，')
        print('对一棵红黑树T，哨兵NIL[T]是一个与树内普通结点具有相同域的对象，', 
            '它的color域为BLACK，其他域的值可以随便设置')
        print('通常将注意力放在红黑树的内部结点上，因为存储了关键字的值')
        print('在本章其余部分，画红黑树时都将忽略其叶子')
        print('从某个结点x出发，到达一个叶子结点的任意一条路径上，', 
            '黑色结点的个数称为该结点的黑高度，用bh(x)表示')
        print('引理13.1 一棵有n个内结点的红黑树的高度至多为2lg(n+1)')
        print(' 红黑树中某一结点x为根的子树中中至少包含2^bh(x)-1个内结点')
        print(' 设h为树的高度，从根到叶结点(不包括根)任意一条简单路径上', 
            '至少有一半的结点至少是黑色的；从而根的黑高度至少是h/2,也即lg(n+1)')
        print('红黑树是许多平衡的查找树中的一种，能保证在最坏情况下，', 
            '基本的动态集合操作SEARCH,PREDECESOR,MINIMUM,MAXIMUM,DELETE,INSERT的时间为O(lgn)')
        print('当给定一个红黑树时，第12章的算法TREE_INSERT和TREE_DELETE的运行时间为O(lgn)')
        print('这两个算法并不直接支持动态集合操作INSERT和DELETE，',
            '但并不能保证被这些操作修改过的二叉查找树是一颗红黑树')
        print('练习13.1-1 红黑树不看颜色只看键值的话也是一棵二叉查找树，只是比较平衡')
        print(' 关键字集合当中有15个元素，所以红黑树的最大黑高度为lg(n+1),n=15,即最大黑高度为4')
        print('练习13.1-2 插入关键字36后，36会成为35的右儿子结点，虽然红黑树中的每个结点的黑高度没有变')
        print(' 如果35是一个红结点，它的右儿子36却是红结点，违反了红黑树性质4，所以插入元素36后不是红黑树')
        print(' 如果35是一个黑结点，则关键字为30的结点直接不满足子路径黑高度相同，所以插入元素36后不是红黑树')
        print('练习13.1-3 定义松弛红黑树为满足性质1,3,4和5，不满足性质2(根结点是黑色的)')
        print(' 也就是说根结点可以是黑色的也可以是红色的，考虑一棵根是红色的松弛红黑树T')
        print(' 如果将T的根部标记为黑色而其他都不变，则所得到的是否还是一棵红黑树')
        print(' 是吧，因为跟结点的颜色改变不影响其子孙结点的黑高度，而根结点自身的颜色与自身的黑高度也没有关系')
        print('练习13.1-4 ')
        print('练习13.1-5 ')
        print('练习13.1-6 ')
        print('练习13.1-7 ')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_2:
    '''
    chpater13.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.2 note

        Example
        ====
        ```python
        Chapter13_2().note()
        ```
        '''
        print('chapter13.2 note as follow')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_3:
    '''
    chpater13.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.3 note

        Example
        ====
        ```python
        Chapter13_3().note()
        ```
        '''
        print('chapter13.3 note as follow')
        print('第13章 ')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

chapter13_1 = Chapter13_1()
chapter13_2 = Chapter13_2()
chapter13_3 = Chapter13_3()

def printchapter13note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter thirteen!')  
    chapter13_1.note()
    chapter13_2.note()
    chapter13_3.note()

# python src/chapter13/chapter13note.py
# python3 src/chapter13/chapter13note.py
if __name__ == '__main__':  
    printchapter13note()
else:
    pass
