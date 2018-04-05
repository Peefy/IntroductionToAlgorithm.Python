
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
        print('设x为二叉查找树中的一个结点，如果y是x的左子树的一个结点，则key[y]<=key[x].')
        print(' 如果y是x的右子树的一个结点，则key[x]<=key[y]')
        print('即二叉查找树的的某结点的左儿子总小于等于自身，右儿子总大于等于自身')
        print('根据二叉查找树的性质，可以用一个递归算法按排列顺序输出树中的所有关键字。')
        print('这种算法成为中序遍历算法，因为一子树根的关键字在输出时介于左子树和右子树的关键字之间')
        print('前序遍历中根的关键字在其左右子树中的关键字输出之前输出，', 
            '而后序遍历中根的关键字再其左右子树中的关键字之后输出')
        print('只要调用INORDER-TREE-WALK(root[T]),就可以输出一棵二叉查找树T中的全部元素')
        print('INORDER-TREE-WALK(x)')
        print('if x != None')
        print('  INORDER-TREE-WALK(left[x])')
        print('  print key[x]')
        print('  INORDER-TREE-WALK(right[x])')
        print('  遍历一棵含有n个结点的二叉树所需的时间为Θ(n),因为在第一次调用遍历过程后，', 
            '对树中的每个结点，该过程都要被递归调用两次')
        print('定理12.1 如果x是一棵包含n个结点的子树的根上，调用INORDER-TREE-WALK过程所需的时间。对于一棵空子树')
        print(' INORDER-TREE-WALK只需很少的一段常量时间(测试x!=None),因而有T(0)=c,c为某一正常数')
        print('练习12.1-1 基于关键字集合{1,4,5,10,16,17,21}画出高度为2,3,4,5,6的二叉查找树')
        print('练习12.1-2 二叉查找树性质与最小堆性质之间有什么区别。能否利用最小堆性质在O(n)时间内，')
        print('  按序输出含有n个结点的树中的所有关键字')
        print('练习12.1-3 给出一个非递归的中序树遍历算法')
        print('  有两种方法，在较为容易的方法中，可以采用栈作为辅助数据结构，在较为复杂的方法中，不采用栈结构')
        print('练习12.1-4 对一棵含有n个结点的树，给出能在Θ(n)时间内，完成前序遍历和后序遍历的递归算法')
        print('练习12.1-5 在比较模型中，最坏情况下排序n个元素的时间为Ω(nlgn),则为从任意的n个元素中构造出一棵二叉查找树')
        print('  任何一个基于比较的算法在最坏情况下，都要花Ω(nlgn)的时间')
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
        print('12.2 查询二叉查找树')
        print('对于二叉查找树，最常见的操作是查找树中的某个关键字。')
        print('除了SEARCH操作外，二叉查找树还能支持诸如MINIMUM,MAXIMUM,SUCCESSOR和PREDECESSOR等查询')
        print('并说明对高度为h的树，它们都可以在O(h)时间内完成')
        print('查找')
        print(' 我们用下面的过程在树中查找一个给定的关键字。给定指向树根的指针和关键字k，', 
            '过程TREE-SEARCH返回包含关键字k的结点(如果存在的话)的指针，否则返回None')
        print('TREE-SEARCH(x,k)')
        print('if x = None or k=key[x]')
        print('  return x')
        print('if k < key[x]')
        print('  return TREE-SEARCH(left[x],k)')
        print('else')
        print('  return TREE-SEARCH(right[x],k)')
        print('最大关键字元素和最小关键字元素')
        print('要查找二叉树中具有最小关键字的元素，只要从根结点开始，沿着各结点的left指针查找下去，直至遇到None时为止')
        print('二叉查找树性质保证了TREE-MINIMUM的正确性。如果一个结点x无子树，', 
            '其右子树中的每个关键字都至少和key[x]一样大')
        print('对高度为h的树，这两个过程的运行时间都是O(h),这是因为，如在TREE-SEARCH过程中一样，', 
            '所遇到的结点序列构成了一条沿着结点向下的路径')
        print('前趋和后继')
        print('给定一个二叉查找树中的结点，有时候要求找出在中序遍历顺序下它的后继')
        print('如果所有的关键字均不相同，则某一结点x的后继即具有大于key[x]中关键字中最小者的那个结点')
        print('根据二叉查找树的结构，不用对关键字做任何比较，就可以找到某个结点的后继')
        print('定理12.2 对一棵高度为h的二叉查找树，动态集合操作SEARCH,MINIMUM,', 
            'MAXIMUM,SUCCESSOR和PREDECESSOR等的运行时间为O(h)')
        print('练习12.2-1 假设在某二叉查找树中，有1到1000之间的一些数，现要找出363这个数。')
        print(' 下列的结点序列中，哪一个不可能是所检查的序列 b),c),e)')
        print('a) 2,252,401,398,330,344,397,363')
        print('b) 924,220,911,244,898,258,362,363')
        print('c) 925,202,911,240,912,245,363')
        print('d) 2,399,387,219,266,382,381,278,363')
        print('e) 935,278,347,621,299,392,358,363')
        print('练习12.2-2 写出TREE-MINIMUM和TREE-MAXIMUM过程的递归版本')
        print('练习12.2-3 写出TREE-PREDECESSOR过程')
        print('练习12.2-4 ')
        print('练习12.2-5 ')
        print('练习12.2-6 ')
        print('练习12.2-7 ')
        print('练习12.2-8 ')
        print('练习12.2-9 ')
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
