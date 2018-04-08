
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

if __name__ == '__main__':
    from searchtree import SearchTree, SearchTreeNode, RandomSearchTree
else:
    from .searchtree import SearchTree, SearchTreeNode, RandomSearchTree

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
        print('练习12.2-4 假设在二叉查找树中，对某关键字k的查找在一个叶结点处结束，考虑三个集合')
        print(' A 包含查找路径左边的关键字；')
        print(' B 包含查找路径上的关键字；')
        print(' C 包含查找路径右边的关键字；')
        print(' 任何三个关键字a∈A,b∈B,c∈C 必定满足a<=b<=c,请给出该命题的一个最小可能的反例')
        print('练习12.2-5 性质：如果二叉查找树中的某结点有两个子女，则其后继没有左子女，其前趋没有右子女')
        print('练习12.2-6 考虑一棵其关键字各不相同的二叉查找树T。证明：如果T中某个结点x的右子树为空，且x有一个后继y')
        print('  那么y就是x的最低祖先，且其左孩子也是x的祖先。')
        print('练习12.2-7 对于一棵包含n个结点的二叉查找树，其中序遍历可以这样来实现；先用TREE-MINIMUM找出树中的最小元素')
        print(' 然后再调用n-1次TREE-SUCCESSOR。证明这个算法的运行时间为Θ(n)')
        print('练习12.2-8 证明：在一棵高度为h的二叉查找树中，无论从哪一个结点开始')
        print(' 连续k次调用TREE-SUCCESSOR所需的时间都是O(k+h)')
        print('练习12.2-9 设T为一棵其关键字均不相同的二叉查找树，并设x为一个叶子结点，y为其父结点。')
        print(' 证明：key[y]或者是T中大于key[x]的最小关键字，或者是T中小于key[x]的最大关键字')
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
        print('12.3 插入和删除')
        print('插入和删除操作会引起二叉查找树表示的动态集合的变化，要反映出这种变化，就要修改数据结构')
        print('但在修改的同时，还要保持二叉查找树性质')
        print('插入一个新元素而修改树的结构相对来说比较简单,但在执行删除操作时情况要复杂一些')
        print('插入：为将一个新值v插入到二叉查找树T中，可以调用TREE-INSERT')
        print(' 传给该过程的参数是个结点z，并且有key[z]=v,left[z]=None,right[z]=None')
        print(' 该过程修改T和z的某些域，并把z插入到树中的合适位置')
        print('定理12.3 对高度为h的二叉查找树，动态集合操作INSERT和DELETE的运行时间为O(h)')
        tree = SearchTree()
        tree.insert_recursive(SearchTreeNode(12, 0))
        tree.insert(SearchTreeNode(11, 1))
        tree.insert(SearchTreeNode(10, 2))
        tree.insert(SearchTreeNode(15, 3))
        tree.insert_recursive(SearchTreeNode(9, 4))   
        print(tree.all())
        print(tree.count())
        print(tree.inorder_tree_walk(tree.root))
        print(tree.tree_search(tree.root, 15))
        print(tree.tree_search(tree.root, 8))
        print(tree.iterative_tree_search(tree.root, 10))
        print(tree.iterative_tree_search(tree.root, 7))
        print(tree.maximum(tree.root))
        print(tree.maximum_recursive(tree.root))
        print(tree.minimum(tree.root))
        print(tree.minimum_recursive(tree.root))
        print(tree.successor(tree.root))
        print(tree.predecessor(tree.root))
        print('练习12.3-1 TREE-INSERT的递归版本测试成功！')
        print('练习12.3-2 假设通过反复插入不同的关键字的做法来构造一棵二叉查找树。论证：为在树中查找一个关键字')
        print(' 所检查的结点数等于插入该关键字所检查的结点数加1')
        print('练习12.3-3 这个排序算法的最好时间和最坏时间:O(h) * n * O(h)', tree.allkey())
        print('练习12.3-4 假设另有一种数据结构中包含指向二叉查找树中某结点y的指针，并假设用过程TREE-DELETE来删除y的前趋z')
        print(' 这样做会出现哪些问题呢，如何改写TREE-DELETE来解决这些问题')
        print('练习12.3-5 删除操作是不可以交换的，先删除x再删除y和先删除y再删除x是不一样的，会影响树的结构')
        print('练习12.3-6 当TREE-DELETE中的结点z有两个子结点时，可以将其前趋(而不是后继)拼接掉')
        print(' 提出了一种公平的策略，即为前趋和后继结点赋予相同的优先级，从而可以得到更好地经验性能。')
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
        print('12.4 随机构造的二叉查找树')
        print('二叉查找树上各基本操作的运行时间都是O(h),h为树的高度。', 
            '但是，随着元素的插入或删除，树的高度会发生变化')
        print('例如各元素是按严格增长的顺序插入的，那么构造出来的树就是一个高度为n-1的链')
        print('要使树的高度尽量平均最小，所以采用随机化技术来随机插入结点')
        print('插入顺序不同则树的结构不同')
        print('如在快速排序中那样，可以证明其平均情况下的行为更接近于最佳情况下的行为，', 
            '而不是接近最坏情况下的行为')
        print('不幸的是，如果在构造二叉查找树时，既用到了插入操作，又用到了删除，那么就很难确定树的平均高度到底是多少')
        print('如果仅用插入操作来构造树，则分析相对容易些。可以定义在n个不同的关键字上的一棵随机构造的二叉查找树')
        print('它是通过按随机的顺序，将各关键字插入一棵初始为空的树而形成的，并且各输入关键字的n!种排列是等可能的')
        print('这一概念不同于假定n个关键字上的每棵二叉查找树都是等可能的')
        print('这一节要证明对于在n个关键字上随机构造的二叉查找树，其期望高度为O(lgn)。假设所有关键字都是不同的')
        print('首先定义三个随机变量，它们有助于测度一棵随机构造的二叉查找树的高度：Xn表示高度')
        print('定义指数高度Yn=2^Xn,Rn表示一个随机变量，存放了该关键字在这n个关键字中的序号')
        print('Rn的值取集合{1,2,...,n}中的任何元素的可能性都是相同的')
        print('定理12.4：一棵在n个关键字上随机构造的二叉查找树的期望高度为O(lgn)')
        random_tree = RandomSearchTree()
        random_tree.randominsertkey(1)
        random_tree.randominsertkey(2)
        random_tree.randominsertkey(3)
        random_tree.randominsertkey(4)
        random_tree.randominsertkey(5)
        random_tree.update()
        random_tree.insertkey(0)
        print(random_tree.all())
        print(random_tree.allkey())
        print(random_tree.inorder_tree_walk(random_tree.root))
        print('练习12.4-1 证明恒等式∑i=0,n-1(i+3, 3)=(n+3, 4)')
        print('练习12.4-2 请描述这样一个的一棵二叉查找树：其中每个结点的平均深度Θ(lgn)，但是树的深度为ω(lgn)')
        print(' 对于一棵含n个结点的二叉查找树，如果其中每个结点的平均深度为Θ(lgn),给出其高度的一个渐进上界O(nlgn)')
        print('练习12.4-3 说明基于n个关键字的随机选择二叉查找树概念(每棵包含n个结点的树被选到的可能性相同)，与本节中介绍的随机构造二叉查找树的概念是不同的')
        print('练习12.4-4 证明f(x)=2^x是凸函数')
        print('练习12.4-5 现对n个输入数调用RANDOMIZED-QUICKSORT。', 
            '证明：对任何常数k>0,输入数的所有n!中排列中，除了其中的O(1/n^k)中排列之外，都有O(nlgn)的运行时间')
        print('思考题12-1 具有相同关键字的二叉查找树')
        print(' 具有相同关键字的存在，给二叉查找树的实现带来了一些问题')
        print(' 当用TREE-INSERT将n个具有相同关键字的数据项插入到一棵初始为空的二叉查找树中，该算法的渐进性能如何')
        print(' 可以对TREE-INSERT做一些改进，即在第5行的前面测试key[z]==key[x],在第11行前面测试key[z]==key[y]')
        print(' 在结点x处设一个布尔标志b[x],并根据b[x]的不同值，置x为left[x]或right[x],', 
            '每当插入一个与x具有相同关键字的结点时，b[x]取TRUE或FALSE,随机地将x置为left[x]或right[x]')
        print('思考题12-2 基数树RadixTree数据结构')
        print(' 给定两个串a=a0a1...ap和b=b0b1...bp,其中每一个ai和每一个bj都属于某有序字符集')
        print(' 例如，如果a和b是位串,则根据规则10100<10110, 10100<101000,这与英语字典中的排序很相似')
        print(' 设S为一组不同的二进制串构成的集合，各串的长度之和为n，说明如何咯用基数树，在Θ(n)时间内将S按字典序排序')
        print('思考题12-3 随机构造的二叉查找树中的平均结点深度')
        print('思考题12-4 证明在一棵随机构造的二叉查找树中，n个结点的平均深度为O(lgn)')
        print(' 与RANDOMIZED-QUICKSORT的运行机制之间的令人惊奇的相似性')
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
