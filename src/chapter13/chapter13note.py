
# python src/chapter13/chapter13note.py
# python3 src/chapter13/chapter13note.py
'''
Class Chapter13_1

Class Chapter13_2

Class Chapter13_3

Class Chapter13_4

'''

from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import sys as _sys
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange

if __name__ == '__main__':
    import redblacktree as rb
else:
    from . import redblacktree as rb

import io
import sys 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 

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
        print('练习13.3-5 考虑用RB-INSERT插入n个结点而成的一棵红黑树。证明：如果n>1,n=2时，则该树至少有一个红结点，',
            '第一个结点就是父节点一定为黑色结点，第二个插入结点则一定为红结点')
        print('练习13.3-6 如果红黑树的表示中不提供父指针的话，应当如何有效地实现RB-INSERT')
        print(' 不会')
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
        print('13.4 删除')
        print('和n个结点的红黑树上的其他基本操作一样，对一个结点的删除要花O(lgn)时间。', 
            '与插入操作相比，删除操作只是稍微复杂些')
        print('程序RB-DELETE是对TREE-DELETE程序(12.3)略作修改得来的。',
            '在删除一个结点后，该程序就调用一个辅助程序RB-DELETE-FIXUP,用来改变结点的颜色并作旋转，从而保持红黑树性质')
        print('过程TREE-DELETE和RB-DELETE之间有三点不同。首先，', 
            'TREE-DELETE中所有对NIL的引用在RB-DELETE中都被替换成哨兵Nil的引用')
        print('如果y是红色的，则当y被删除后，红黑性质仍然得以保持，理由如下：')
        print(' 树中各结点的黑高度都没有变化')
        print(' 不存在两个相邻的红色结点')
        print(' 因为如果y是红的，就不可能是根，所以根仍然是黑色的')
        print('传递给RB-DELETE-FIXUP的结点x是两个结点中的一个：在y被删除之前，如果y有个不是哨兵nil的孩子')
        print('则x为y的唯一孩子；如果y没有孩子，则x为哨兵nil,在后一种情况中，',
            '之后的无条件赋值保证了无论x是有关键字的内结点或哨兵nil,x现在的父结点都为先前y的父结点')
        print('在RB-DELETE中，如果被删除的结点y是黑色的，则会产生三个问题。')
        print(' 首先，如果y原来是根结点，而y的一个红色的孩子成为了新的根，就会违反性质2')
        print(' 其次，如果x和y.p都是红的，就会违反性质4')
        print(' 第三，删除y将导致先前包含y的任何路径上黑结点的个数少1，因此性质5被y的一个祖先破坏')
        print('  补救这个问题的一个办法就是把结点x视为还有额外的一重黑色。')
        print('RB-DELETE-FIXUP程序负责恢复性质1,2,4')
        print('while循环的目标是将额外的黑色沿树上移动,直到')
        print(' 1.x指向一个红黑结点，此时将x着色为黑色')
        print(' 2.x指向根，这是可以简单地消除那个额外的黑色，或者')
        print(' 3.做必要的旋转和颜色修改')
        print('RB-DELETE-FIXUP程序的几种情况')
        print(' 情况1.x的兄弟w是红色的')
        print(' 情况2.x的兄弟w是黑色的，而且w的两个孩子都是黑色的')
        print(' 情况3.x的兄弟w是黑色的，w的左孩子是红色的，右孩子是黑色的')
        print(' 情况4.x的兄弟w是黑色的，而且w的右孩子是红色的')
        print('RB-DELETE的运行时间：含n个结点的红黑树的高度为O(lgn),',
            '不调用RB-DELETE-FIXUP时该程序的总时间代价为O(lgn)')
        print('在RB-DELETE-FIXUP中，情况1,3和4在各执行一定次数的颜色修改和至多修改三次旋转后便结束')
        print('情况2是while循环唯一可以重复的情况，其中指针x沿树上升的次数至多为O(lgn)次，且不执行任何旋转')
        print('所以，过程RB-DELETE-FIXUP要花O(lgn)时间，做至多三次旋转，从而RB-DELETE的总时间为O(lgn)')
        print('练习13.4-1: 红黑树删除过程性质：在执行RB-DELETE-FIXUP之后，树根总是黑色的')
        print('练习13.4-2: 在RB-DELETE中，如果x和p[y]都是红色的，则性质4可以通过调用RB-DELETE-FIXUP(T,x)来恢复')
        print('练习13.4-3: 在练习13.3-2中，将关键字41,38,31,12,19,8连续插入一棵初始为空的树中，从而得到一棵红黑树。')
        print(' 请给出从该树中连续删除关键字8,12,19,31,38,41后的结果')
        tree = rb.RedBlackTree()
        nodekeys = [41, 38, 31, 12, 19, 8]
        for key in nodekeys:
            tree.insertkey(key)
        print(tree.all())
        nodekeys.reverse()
        for key in nodekeys:
            tree.deletekey(key)
        print(tree.all())
        tree.insertkey(1)
        print(tree.all())
        print('练习13.4-4: 在RB-DELETE-FIXUP的哪些行中，可能会检查或修改哨兵nil[T]?')
        print('练习13.4-5: ')
        print('练习13.4-6: x.p在情况1的开头一定是黑色的')
        print('练习13.4-7: 假设用RB-INSERT来将一个结点x插入一棵红黑树，紧接着又用RB-DELETE将它从树中删除')
        print(' 结果的红黑树与初始的红黑树是否相同？')
        print('思考题13-1: 持久动态集合')
        print(' 在算法的执行过程，会发现在更新一个动态集合时，需要维护其过去的版本，这样的集合被称为是持久的')
        print(' 实现持久集合的一种方法是每当该集合被修改时，就将其整个地复制下来，但是这种方法会降低一个程序的执行速度，而且占用过多的空间。')
        print(' 考虑一个有INSERT，DELETE和SEARCH操作的持久集合S，对集合的每一个版本都维护一个单独的根')
        print(' 为把关键字5插入的集合中去，就要创建一个具有关键字5的新结点,最终只是复制了树的一部分，新树和老树之间共享一些结点')
        print(' 假设树中每个结点都有域key,left,right,但是没有父结点的域')
        print(' 1.对一棵一般的持久二叉查找树，为插入一个关键字k或删除一个结点y，确定需要改变哪些结点')
        print(' 2.请写出一个程序PERSISTENT-TREE-INSERT,使得在给定一棵持久树T和一个要插入的关键字k时，它返回将k插入T后新的持久树T1')
        print(' 3.如果持久二叉查找树T的高度为h，所实现的PERSISTENT-TREE-INSERT的时间和空间要求分别是多少')
        print(' 4.假设我们在每个结点中增加一个父亲结点域。这样一来：PERSISTENT-TREE-INSERT需要做一些额外的复制工作')
        print('  证明在这种情况下。PERSISTENT-TREE-INSERT的时空要求Ω(n),其中n为树中的结点个数')
        print(' 5.说明如何利用红黑树来保证每次插入或删除的最坏情况运行时间为O(lgn)')
        print('思考题13-2: 红黑树上的连接操作')
        print(' 连接操作以两个动态集合S1和S2和一个元素x为参数，使对任何x1属于S1和x2属于S2')
        print(' 有key[x1]<=key[x]<=key[x2],该操作返回一个集合S=S1 ∪ {x} ∪ S2。')
        print(' 在这个问题中，讨论在红黑树上实现连接操作')
        print(' 1.给定一棵红黑树T，其黑高度被存放在域bh[T]。证明不需要树中结点的额外存储空间和', 
            '不增加渐进运行时间的前提下，可以用RB-INSERT和RB-DELETE来维护这个域')
        print(' 希望实现RB-JOIN(T1,x,T2),它删除T1和T2，并返回一棵红黑树T= T1 ∪ {x} ∪ T2')
        print(' 设n为T1和T2中的总结点数')
        print(' 证明RB-JOIN的运行时间是O(lgn)')
        print('思考题13-3: AVL树是一种高度平衡的二叉查找树:对每一个结点x，x的左子树与右子树的高度至多为1')
        print(' 要实现一棵AVL树，我们在每个结点内维护一个额外的域:h(x),即结点的高度。至于任何其他的二叉查找树T，假设root[T]指向根结点')
        print(' 1.证明一棵有n个结点的AVL树其高度为O(lgn)。证明在一个高度为h的AVL树中，至少有Fh个结点，其中Fh是h个斐波那契数')
        print(' 2.为把结点插入到一棵AVL树中，首先以二叉查找树的顺序把结点放在适当的位置上')
        print('  这棵树可能就不再是高度平衡了。具体地，某些结点的左子树与右子树的高度差可能会到2')
        print('  请描述一个程序BALANCE(x),输入一棵以x为根的子树，其左子树与右子树都是高度平衡的，而且它们的高度差至多是2')
        print('  即|h[right[x]]-h[left[x]]|<=2,然后将以x为根的子树转变为高度平衡的')
        print(' 3.请给出一个由n个结点的AVL树的例子，其中一个AVL-INSERT操作将执行Ω(lgn)次旋转')
        print('思考题13-4: Treap')
        print(' 如果将一个含n个元素的集合插入到一棵二叉查找树中，所得到的树可能会非常不平衡，从而导致查找时间很长')
        print(' 随机构造的二叉查找树往往是平衡的。因此，一般来说，要为一组固定的元素建立一棵平衡树，可以采用的一种策略')
        print(' 就是先随机排列这些元素，然后按照排列的顺序将它们插入到树中')
        print(' 如果一次收到一个元素，也可以用它们来随机建立一棵二叉查找树')
        print(' 一棵treap是一棵修改了结点顺序的二叉查找树。')
        print(' 通常树内的每个结点x都有一个关键字值key[x]。')
        print(' 另外，还要为结点分配priority[x],它是一个独立选取的随机数。假设所有的优先级都是不同的')
        print(' 在将一个结点插入一棵treap树内时，所执行的旋转期望次数小于2')
        print('AVL树：最早的平衡二叉树之一。应用相对其他数据结构比较少，windows对进程地址空间的管理用到了AVL树,平衡度也最好')
        print('红黑树：平衡二叉树，广泛应用在C++的STL中。如map和set都是用红黑树实现的')
        print('B/B+树：用在磁盘文件组织，数据索引和数据库索引')
        print('Trie树(字典树)：用在统计和排序大量字符串，如自动机')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

chapter13_1 = Chapter13_1()
chapter13_2 = Chapter13_2()
chapter13_3 = Chapter13_3()
chapter13_4 = Chapter13_4()

def printchapter13note():
    '''
    print chapter13 note.
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
