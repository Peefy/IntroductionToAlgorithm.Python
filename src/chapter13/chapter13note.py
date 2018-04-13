
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

if __name__ == '__main__':
    import redblacktree as rb
else:
    from . import redblacktree as rb

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
        print('练习13.1-4 假设一棵红黑树的每一个红结点\"结点\"吸收到它的黑色父节点中，来让红结点的子女变成黑色父节点的子女')
        print(' 其可能的度是多少，次结果树的叶子深度怎样，因为红黑树中根的黑高度至少是h/2，红结点都被吸收的话，叶子结点变为原来的一半')
        print('练习13.1-5 红黑树一个定理：在一棵红黑树中，从某结点x到其后代叶子结点的所有简单路径中，最长的一条是最短一条的至多两倍')
        print('练习13.1-6 因为一棵有n个内结点的红黑树的高度至多为2lg(n+1),则高度为k，内结点个数最多为[2^(k/2)]-1')
        print('练习13.1-7 请描述出一棵在n个关键字上构造出来的红黑树，使其中红的内结点数与黑的内结点数的比值最大')
        print(' 这个比值是多少，具有最小可能比例的树又是怎样？此比值是多少')
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
        print('13.2 旋转')
        print('当在含n个关键字的红黑树上运行时，查找树操作TREE-INSERT和TREE-DELETE的时间为O(lgn)')
        print('由于这两个操作对树作了修改，结果可能违反13.1节中给出的红黑性质。',
            '为保持这些性质，就要改变树中某些结点的颜色以及指针结构')
        print('指针结构的修改是通过旋转来完成的，这是一种能保持二叉查找树性质的查找树局部操作')
        print('给出左旋和右旋。当某个结点x上做左旋时，假设它的右孩子不是nil[T],',
            'x可以为树内任意右孩子不是nil[T]的结点')
        print('左旋以x到y之间的链为\"支轴\"进行，它使y成为该该子树新的根，x成为y的左孩子，而y的左孩子则成为x的右孩子')
        print('在LEFT-ROTATE的代码中，必须保证right[x]!=None,且根的父结点为None')
        print('练习13.2-1 RIGHT-ROTATE的代码已经给出')
        print('练习13.2-2 二查查找树性质：在一棵有n个结点的二叉查找树中，刚好有n-1种可能的旋转')
        print('练习13.2-3 属于x结点的子孙结点，当结点x左旋时，x的子孙结点的深度加1')
        print('练习13.2-4 二查查找树性质：任何一棵含有n个结点的二叉查找树，可以通过O(n)次旋转，',
            '转变为另外一棵含n个结点的二叉查找树')
        print('练习13.2-5 如果二叉查找树T1可以右转成二叉查找树T2，则可以调用O(n^2)次RIGHT-ROTATE来右转')
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
        print('13.3 插入')
        print('一棵含有n个结点的红黑树中插入一个新结点的操作可以在O(lgn)时间内完成')
        print('红黑树T插入结点z时，就像是T是一棵普通的二叉查找树那样，然后将z染成红色，')
        print('为保证红黑性质能继续保持，调用一个辅助程序对结点重新旋转和染色，假设z的key域已经提前赋值')
        print('插入结点后，如果有红黑性质被破坏，则至多有一个被破坏，并且不是性质2就是性质4')
        print('如果违反性质2，则发生的原因是z的根而且是红的，')
        print('如果违反性质4，则原因是z和z的父结点都是红的')
        print('循环结束是因为z.p是黑的，')
        print('在while循环中需要考虑六种情况，其中三种与另外三种')
        print('情况1与情况2，3的区别在于z的父亲的兄弟(或叔叔)的颜色有所不同，')
        print('情况1:z的叔叔y是红色的')
        print('情况2:z的叔叔y是黑色的，而且z是右孩子')
        print('情况3:z的叔叔y是黑色的，而且z是左孩子')
        print('有趣的是，insert_fixup的整个过程旋转的次数从不超过两次')
        tree = rb.RedBlackTree()
        tree.insertkey(41)
        tree.insertkey(38)
        tree.insertkey(31)
        tree.insertkey(12)
        tree.insertkey(19)
        tree.insertkey(8)
        tree.insertkey(1)
        print('练习13.3-1 红黑树假设插入的结点x是红色的，但是将结点假设为黑色，则红黑树的性质4就不会破坏')
        print(' 但是不会这么做，原因是会直接改变其父亲结点的黑高度，破坏红黑树的性质5，这样会使红黑树的插入变得非常复杂')
        print('练习13.3-2 ', tree.inorder_tree_walk(tree.root))
        print('练习13.3-3 略')
        print('练习13.3-4 红黑树性质：RB-INSERT-FIXUP过程永远不会将color[nil[T]]设置为RED')
        print('练习13.3-5 ')
        print('练习13.3-6 ')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_4:
    '''
    chpater13.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.4 note

        Example
        ====
        ```python
        Chapter13_4().note()
        ```
        '''
        print('chapter13.4 note as follow')
        print('13.4 插入')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

chapter13_1 = Chapter13_1()
chapter13_2 = Chapter13_2()
chapter13_3 = Chapter13_3()
chapter13_4 = Chapter13_4()

def printchapter13note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter thirteen!')  
    chapter13_1.note()
    chapter13_2.note()
    chapter13_3.note()
    chapter13_4.note()

# python src/chapter13/chapter13note.py
# python3 src/chapter13/chapter13note.py
if __name__ == '__main__':  
    printchapter13note()
else:
    pass
