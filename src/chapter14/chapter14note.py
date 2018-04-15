
# python src/chapter14/chapter14note.py
# python3 src/chapter14/chapter14note.py
'''
Class Chapter14_1

Class Chapter14_2

Class Chapter14_3

Class Chapter14_4

'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter14_1:
    '''
    chpater14.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.1 note

        Example
        ====
        ```python
        Chapter14_1().note()
        ```
        '''
        print('chapter14.1 note as follow')
        print('第14章 数据结构的扩张')
        print('在有些工程应用环境中，需要一些标准的数据结构，如双链表、散列表或二叉查找树')
        print('同时，也有许多应用要求在现有数据结构上有所创新，但很少需要长早出全新的数据结构')
        print('这一章讨论两种通过扩充红黑树构造的数据结构')
        print('14.1 动态顺序统计')
        print('第9章中介绍了顺序统计的概念。例如，在包含n个元素的集合中，第i个顺序统计量即为该集合中具有第i小关键字的元素')
        print('在一个无序的集合中，任意的顺序统计量都可以在O(n)时间内找到')
        print('这一节里，将介绍如何修改红黑树的结构，使得任意的顺序统计量都可以在O(lgn)时间内确定')
        print('还将看到，一个元素的排序可以同样地在O(lgn)时间内确定')
        print('一棵顺序统计量树T通过简单地在红黑树的每个结点存入附加信息而成',
            '在一个结点x内，除了包含通常的红黑树的域key[x],color[x],p[x],left[x]和right[x],还包括域size[x]')
        print('这个域中包含以结点x为根的子树的内部结点数(包括x本身)，即子树的大小，如果定义哨兵为0，也就是设置size[nil[T]]为0')
        print('则有等式size[x]=size[left[x]]+size[right[x]]+1')
        print('在一个顺序统计树中，并不要求关键字互不相同')
        print('在出现相等关键字的情况下，先前排序的定义不再适用。')
        print('定义排序为按中序遍历树时输出的结点位置，以此消除顺序统计树原定义的不确定性')
        print('OS-RANK的while循环的每一次迭代要花O(1)时间，且y在每次迭代中沿树上升一层')
        print(' 所以最坏情况下，OS-RANK的运行时间与树的高度成正比：对含n个结点的顺序统计树时间为O(lgn)')
        print('对子树规模的维护：给定每个结点的size域后，OS-SELECT和OS-RANK能迅速计算出所需的顺序统计信息')
        print('维护size域的代价为O(lgn)')
        print('红黑树上的插入操作包括两个阶段。第一个阶段从根开始，沿着树下降，将新结点插入为某个已有结点的孩子')
        print('第二阶段沿树上升，做一些颜色修改和旋转以保持红黑性质')
        print('于是，向一个含n个结点的顺序统计树中插入所需的总时间为O(lgn),从渐进意义上来看，这与一般的红黑树是相同的')
        print('红黑树上的删除操作同样包含两个阶段：第一阶段对查找树进行操作；第二阶段做至多三次旋转')
        print('综上所述,插入操作和删除操作，包括维护size域，都需O(lgn)时间')
        print('练习14.1-1: 略')
        print('练习14.1-2: 略')
        print('练习14.1-3: 完成')
        print('练习14.1-4: 完成')
        print('练习14.1-5: 给定含n个元素的顺序统计树中的一个元素x和一个自然数i，',
            '如何在O(lgn)时间内，确定x在该树的线性序中第i个后继')
        print('练习14.1-6: 在OS-SELECT或OS-RANK中，每次引用结点的size域都',
            '仅是为了计算在以结点为根的子树中该结点的rank')
        print('练习14.1-7: 说明如何在O(nlgn)的时间内，利用顺序统计树对大小为n的数组中的逆序对')
        print('练习14.1-8: 现有一个圆上的n条弦，每条弦都是按其端点来定义的。',
            '请给出一个能在O(nlgn)时间内确定园内相交弦的对数。假设任意两条弦都不会共享端点')      
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

class Chapter14_2:
    '''
    chpater14.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.2 note

        Example
        ====
        ```python
        Chapter14_2().note()
        ```
        '''
        print('chapter14.2 note as follow')
        print('14.2 如何扩张数据结构')
        print('对一种数据结构的扩张过程可分为四个步骤')
        print(' 1.选择基础数据结构')
        print(' 2.确定要在基础数据结构中添加哪些信息')
        print(' 3.验证可用基础数据结构上的基本修改操作来维护这些新添加的信息')
        print(' 4.设计新的操作')
        print('对红黑树的扩张')
        print('当红黑树被选作基础数据结构时，可以证明，',
            '某些类型的附加信息总是可以用插入和删除来进行有效地维护')
        print('定理14.1(红黑树的扩张) 设域f对含n个结点的红黑树进行扩张的域，',
            '且某结点x的域f的内容可以仅用结点x,left[x]和right[x]中的信息计算')
        print(' 包括f[left[x]]和f[right[x]]')
        print(' 在插入和删除操作中，我们在不影响这两个操作O(lgn)渐进性能的情况下，对T的所有结点的f值进行维护')
        print('练习14.2-1 通过为结点增加指针的形式可以在扩张的顺序统计树上，以最坏情况O(1)的时间来支持')
        print(' 动态集合查询操作MINIMUM,MAXIMUM,SUCCESSOR,PREDECESSOR,且顺序统计树上的其他操作的渐进性能不受影响')
        print('练习14.2-2 可以为结点增加黑高度域，且不影响红黑树操作性能')
        print('练习14.2-3 可以将红黑树中结点的深度作为一个域来进行有效的维护')
        print('练习14.2-4 假设在红黑树每个结点x中增加一个域f，按中序排列所有结点。证明在一次旋转后')
        print(' 可以在O(1)时间里对f域作出合适的修改。对扩张稍作修改，证明顺序统计树size域的每次旋转的维护时间为O(1)')
        print('练习14.2-5 希望通过增加操作RB-ENUMERATE(x,a,b)来扩张红黑树。该操作输出所有的关键字k')
        print(' 在Θ(m+lgn)时间里实现RB-ENUMERATE，其中m为输出的关键字数，n为树中内部结点(不需要向红黑树中增加新的域)')
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

class Chapter14_3:
    '''
    chpater14.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.3 note

        Example
        ====
        ```python
        Chapter14_3().note()
        ```
        '''
        print('chapter14.3 note as follow')
        print('14.3 区间树')
        print('定义一个闭区间是一个有序对[t1,t2]。开区间和半开区间分别略去了集合的两个或一个端点')
        print('区间可以很方便地表示各占用一段连续时间的一些事件')
        print('区间树是一种对动态集合进行维护的红黑树，该集合中的每个元素x都包含一个区间int[x]')
        print('区间树以及区间树上各种操作的设计')
        print(' 1.基础数据结构：红黑树；其中，每个结点x包含一个区间域int[x],x的关键字为区间的低端点low[int[x]]')
        print(' 2.附加信息：每个结点中除了区间信息外，还包含一个值max[x],即以x为根的子树中所有区间的端点的最大值')
        print(' 3.对信息的维护:必须验证对含n个结点的区间树的插入和删除能在O(lgn)时间内完成')
        print('  给定区间int[x]和x的子结点的max值，可以确定max[x]')
        print('     max[x]=max(x.high, x.left.max, x.right.max)')
        print('  这样根据定理14.1可知，插入和删除操作的运行时间O(lgn)。在一次旋转后，更新max域只需O(1)时间')
        print(' 4.设计新的操作:唯一需要的新操作是INTERVAL-SEARCH(T, i)')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

class Chapter14_4:
    '''
    chpater14.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.4 note

        Example
        ====
        ```python
        Chapter14_4().note()
        ```
        '''
        print('chapter14.4 note as follow')
        print('')
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

chapter14_1 = Chapter14_1()
chapter14_2 = Chapter14_2()
chapter14_3 = Chapter14_3()
chapter14_4 = Chapter14_4()

def printchapter14note():
    '''
    print chapter14 note.
    '''
    print('Run main : single chapter fourteen!')  
    chapter14_1.note()
    chapter14_2.note()
    chapter14_3.note()
    chapter14_4.note()

# python src/chapter14/chapter14note.py
# python3 src/chapter14/chapter14note.py
if __name__ == '__main__':  
    printchapter14note()
else:
    pass
