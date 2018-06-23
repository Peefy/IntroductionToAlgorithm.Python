# coding:utf-8
# usr/bin/python3
# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
'''

Class Chapter22_1

Class Chapter22_2

Class Chapter22_3

Class Chapter22_4

Class Chapter22_5

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

if __name__ == '__main__':
    import graph as _g
else: 
    from . import graph as _g

class Chapter22_1:
    '''
    chpater22.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.1 note

        Example
        ====
        ```python
        Chapter22_1().note()
        ```
        '''
        print('chapter22.1 note as follow')  
        print('第六部分 图算法')
        print('图是计算机科学中常用的一类数据结构，有关图的算法也是计算机科学中基础性算法')
        print('有许多有趣而定计算问题都是用图来定义的')
        print('第22章介绍图在计算机中的表示，并讨论基于广度优先或深度优先图搜索的算法')
        print(' 给出两种深度优先搜索的应用；根据拓扑结构对有向无回路图进行排序，以及将有向图分解为强连通子图')
        print('第23章介绍如何求图的最小权生成树(minimum-weight spanning tree)')
        print(' 定义：即当图中的每一条边都有一个相关的权值时，',
            '这种树由连接了图中所有顶点的、且权值最小的路径所构成')
        print(' 计算最小生成树的算法是贪心算法的很好的例子')
        print('第24章和第25章考虑当图中的每条边都有一个相关的长度或者\"权重\"时，如何计算顶点之间的最短路径问题')
        print('第24章讨论如何计算从一个给定的源顶点至所有其他顶点的最短路径问题')
        print('第25章考虑每一对顶点之间最短路径的计算问题')
        print('第26章介绍在物流网络(有向图)中，物流的最大流量计算问题')
        print('在描述某一给定图G=(V, E)上的一个图算法的运行时间，通常以图中的顶点个数|V|和边数|E|来度量输入规模')
        print('比如可以讲该算法的运行时间为O(VE)')
        print('用V[G]表示一个图G的顶点集,用E[G]表示其边集')
        print('第22章 图的基本算法')
        print('22.1 图的表示')
        print('要表示一个图G=(V,E),有两种标准的方法，即邻接表和邻接矩阵')
        print('这两种表示法即可以用于有向图，也可以用于无向图')
        print('通常采用邻接表表示法，因为用这种方法表示稀疏图比较紧凑')
        print('但是，当遇到稠密图(|E|接近于|V|^2)或必须很快判别两个给定顶点是否存在连接边，通常采用邻接矩阵表示法')
        print('图G=(V,E)的邻接表表示由一个包含|V|个列表的数组Adj所组成,其中每个列表对应于V中的一个顶点')
        print('对于每一个u∈V，邻接表Adj[u]包含所有满足条件(u,v)∈E的顶点v')
        print('亦即Adj[u]包含图G中所有的顶点u相邻的顶点')
        print('如果G是一个有向图,则所有邻接表的长度之和为|E|,这是因为一条形如',
            '(u,v)的边是通过让v出现在Adj[u]中来表示的')
        print('如果G是一个无向图，则所有邻接表的长度之和为2|E|')
        print('因为如果(u,v)是一条无向边,那么u就会出现在v的邻接表中')
        print('不论是有向图还是无向图，邻接表表示法都有一个很好的特性，即它所需要的存储空间为Θ(V+E)')
        print('邻接表稍作变动，即可用来表示加权图，即每条边都有着相应权值的图')
        print('权值通常由加权函数w给出，例如设G=(V,E)是一个加权函数为w的加权图')
        print('邻接表表示法稍作修改就能支持其他多种图的变体，因而有着很强的适应性')
        print('邻接表表示法也有着潜在不足之处，即如果要确定图中边(u,v)是否存在，',
            '只能在顶点u的邻接表Adj[u]中搜索v,除此之外，没有其他更快的方法')
        print('这个不足可以通过图的邻接矩阵表示法来弥补，但要在(在渐进意义下)以占用更多的存储空间作为代价')
        # !一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的
        print('一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的')
        print('邻接矩阵是沿主对角线对称的')
        print('正如图的邻接表表示一样，邻接矩阵也可以用来表示加权图')
        print('例如，如果G=(V,E)是一个加权图，其权值函数为w，对于边(u,v)∈E,其权值w(u,v)')
        print('就可以简单地存储在邻接矩阵的第u行第v列的元素中，如果边不存在，则可以在矩阵的相应元素中存储一个None值')
        # !邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的
        print('邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的')
        print('但由于邻接矩阵简单明了,因而当图较小时,更多地采用邻接矩阵来表示')
        print('另外如果一个图不是加权的，采用邻接军阵的存储形式还有一个优越性:',
            '在存储邻接矩阵的每个元素时，可以只用一个二进制位，而不必用一个字的空间')
        print('练习22.1-1 给定一个有向图的邻接表示，计算该图中每个顶点的出度和入度都为O(V+E)')
        print(' 计算出度和入度的过程相当于将邻接链表的顶点和边遍历一遍')
        print('练习22.1-2 给出包含7个顶点的完全二叉树的邻接表表示，写出其等价的邻接矩阵表示')
        g = _g.Graph()
        g.veterxs = ['1', '2', '3', '4', '5', '6', '7']
        g.edges = [('1', '2'), ('1', '3'), ('2', '4'),
               ('2', '5'), ('3', '6'), ('3', '7')]
        print(g.getmatrix())
        print('练习22.1-3 邻接链表：对于G的每个节点i，遍历；adj,将i添加到adj中遇到的每个结点')
        print(' 时间就是遍历邻接链表的时间O(V+E)')
        print('邻接矩阵：就是求G的转置矩阵，时间为O(V^2)')
        print('练习22.1-4 给定一个多重图G=(V,E)的邻接表表示,给出一个具有O(V+E)时间的算法,',
            '计算“等价”的无向图G`(V,E`)的邻接表，其中E`包含E中所有的边,',
            '且将两个顶点之间的所有多重边用一条边代表，并去掉E中所有的环')
        print('练习22.1-5 算法运行时间都为O(V^3)')
        print('练习22.1-6 当采用邻接矩阵表示时，大多数图算法需要的时间都是Ω(V^2),但也有一些例外')
        print(' 证明：在给定了一个有向图G邻接矩阵后，可以在O(V)时间内，',
            '确定G中是否包含一个通用的汇，即入度|V|-1,出度为0顶点')
        print('练习22.1-7 矩阵乘积对角线上的元素表示与某结点连接的边的个数')
        print(' 若第m行第n列的元素为-1，则第m个结点到第n个结点连通，并且方向从m到n')
        print('练习22.1-8 假设每个数组元素Adj[u]采用的不是链表,而是一个包含了所有满足(u,v)∈E的顶点v的散列表')
        print(' 如果所有的边查找都是等可能的，则确定某条边是否在途中所需的期望时间是多少')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_2:
    '''
    chpater22.2 note and function
    '''
    def buildGraph(self):
        '''
        练习22.2-1

        练习22.2-2
        '''
        g = _g.Graph()
        g.veterxs = [_g.Vertex('1'), _g.Vertex('2'),
                     _g.Vertex('3'), _g.Vertex('4'),
                     _g.Vertex('5'), _g.Vertex('6')]
        g.edges.clear()
        g.edges.append(_g.Edge(_g.Vertex('1'), _g.Vertex('2'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('4'), _g.Vertex('2'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('1'), _g.Vertex('4'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('2'), _g.Vertex('5'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('3'), _g.Vertex('6'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('3'), _g.Vertex('5'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('5'), _g.Vertex('4'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('6'), _g.Vertex('6'), 1, _g.DIRECTION_TO))
        _g.bfs(g, g.veterxs[2])
        _g.print_path(g, g.veterxs[2], g.veterxs[4])
        print('')
        del g
        
        g = _g.Graph()
        g.veterxs.clear()
        g.edges.clear()
        v = ['r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        g.addvertex(v)
        g.addedge('v', 'r')
        g.addedge('r', 's')
        g.addedge('s', 'w')
        g.addedge('w', 'x')
        g.addedge('w', 't')
        g.addedge('x', 't')
        g.addedge('x', 'u')
        g.addedge('x', 'y')
        g.addedge('y', 'u')
        g.addedge('u', 't')
        _g.bfs(g, 'u')
        _g.print_path(g, 'u', 'v')
        print('')
        del g

    def note(self):
        '''
        Summary
        ====
        Print chapter22.2 note

        Example
        ====
        ```python
        Chapter22_2().note()
        ```
        '''
        print('chapter22.2 note as follow')
        print('22.2 广度优先搜索')
        # !广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型
        print('广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型')
        print('在Prim最小生成树算法和Dijkstra单源最短路径算法，都采用了与广度优先搜索类似的思想')
        print('在给定图G=(V,E)和特定的一个源顶点s的情况下,广度优先搜索系统地探索G中的边，',
            '以期发现可从s到达的所有顶点,并计算s到所有这些可达顶点之间的距离(即最少的边数)')
        print('该搜素算法同时还能生成一棵根为s,且包括所有s的可达顶点的广度优先树')
        print('对从s可达的任意顶点v,从优先树中从s到v的路径对应于图G中从s到v的一条最短路径,',
            '即包含最少边数的路径.该算法对有向图和无向图同样适用')
        print('之所以称为广度优先搜索：算法首先会发现和s距离为k的所有顶点,',
            '然后才会发现和s距离为k+1的其他顶点')
        print('为了记录搜索的轨迹，广度优先搜索将每个顶点都着色为白色，灰色或者黑色')
        print('白色表示还未搜索，灰色和黑色表示已经被发现')
        print('与黑色结点相邻的所有顶点都是已经被发现的。')
        print('灰色顶点可能会有一些白色的相邻结点,它们代表了已经发现与未发现顶点之间的边界')
        print('广度优先搜索构造了一棵广度优先树,在开始时只包含一个根顶点，即源顶点s')
        print('在扫描某个已发现顶点u的邻接表的过程中,每当发现一个白色顶点v,',
            '该顶点v以及边(u,v)就被添加到树中')
        print('在广度优先树中,称u是v的先辈或者父母。',
            '由于一个顶点至多只能被发现一次,因此它最多只能有一个父母顶点。',
            '在广度优先树中，祖先和后裔关系的定义和通常一样,是相对于根s来定义的：',
            '如果u处于树中从根s到顶点v的路径中，那么u称为v的祖先，v是u的后裔')
        _g.test_bfs()
        print('只要队列Q中还有灰色顶点(即那些已经被发现，但是还没有)')
        print('广度优先搜索运行时间分析')
        print(' 采用聚集分析技术,由于所有邻接表长度之和为Θ(E).初始化操作的开销为O(V)')
        print(' 过程BFS的总运行时间为O(V+E),由此可见，',
            '广度优先搜索的运行时间是图G的邻接表大小的一个线性函数')
        print('最短路径')
        print(' 对于一个图G=(V,E),(有向图和无向图均可以),广度优先搜索算法可以得到从已知源顶点s∈V到每个可达顶点的距离')
        print(' 定义从顶点s到v之间的最短路径距离d(s,v)为s到v的任何路径中最少的边数')
        print('  如果两个点s到v之间没有同路，则距离为无穷')
        print(' 广度优先搜索计算出来的就是最短路径')
        print('引理22.1 设G=(V,E)是一个有向图或无向图,s∈V为G的任意一个顶点，则对任意边(u,v)∈V,有:d(s,v)<=d(s,u)+1')
        print('证明：如果从顶点s可达顶点u，则从s也可达v.在这种情况下，从s到v的最短路径不可能比s到u的最短路径加上边(u,v)更长',
            '因此不等式成立。如果从s不可达顶点u，则d(s,u)=∞，不等式仍然成立')
        print('引理22.2 设G=(V,E)是一个有向图或无向图,并假设算法BFS(广度优先搜索)从G中某一给定源顶点s∈V开始执行',
            '在执行终止时，对每个顶点v∈V，BFS所计算出来的v.d的值没看组v.d>=d(s,v)')
        print('引理22.3 假设过程BFS在图G=(V,E)上的执行过程中,队列Q包含顶点<v1,v2,..,vr>',
            '其中v1是队列的头，vr是队列的尾巴','则d[vr]<=d[v1]+1,i=1,2,...,r-1')
        print('推论22.4 假设在BFS的执行过程中将顶点vi和vj插入了队列，且vi先于vj入队',
            '那么，当vj入队时，有d[vi]<=d[vj]')
        print('定理22.5 (广度优先搜索的正确性)设G=(V,E)是一个有向图或无向图，',
            '并假设过程BFS从G上某个给定的源顶点s可达的每一个顶点v∈V。在运行终止时，对所有v∈V，',
            'd[v]=d(s,v).此外，对任意从s可达的顶点v≠s,从s到v的最短路径之一是从s到v.pi的最短路径再加上边(v.pi,v)')
        print('广度优先树')
        print('过程BFS在搜索图的同时，也建立了一棵广度优先树。这棵树是由每个顶点中的pi域所表示的')
        print('对于图G=(V,E)及给定的源顶点s，可以更为形式化地定义其前趋子图Gpi=(Vpi,Epi)')
        print('引理22.6 当过程BFS应用于某一有向图或无向图G=(V,E)时，',
              '同时要构造出pi域,使得前趋子图Gpi=(Vpi,Epi)是一棵广度优先树')
        print('PRINT-PATH(G,s,v)过程输出从s到v的最短路径上的所有结点',
              '假设已经运行了BFS来计算最短路径')
        print('练习22.2-1 ')
        print('练习22.2-2 ')
        self.buildGraph()
        print('练习22.2-3 略')
        print('练习22.2-4 在广度优先搜索算法BFS中,赋给顶点u的值d[u]与顶点在邻接表中的次序无关')
        print(' 由BFS计算出来的广度优先树与邻接表中的顺序是有关的')
        print('练习22.2-5 在有向图G=(V,E)中，源顶点s∈V，且树边集合满足对每一顶点v∈V，',
            '从s到v的唯一路径是G中的一条最短路径;然而不论在每个邻接表中各顶点如何排列，',
            '都不能通过在G上运行BFS而产生边集')
        print('练习22.2-6 略')
        print('练习22.2-7 树T=(V,E)的直径定义为max(d(u,v)),亦即，树的直径是树中所有最短路径长度中的最大值',
            '试写出计算树的直径的有效算法，并分析算法的运行时间')
        print(' 用无向图构造树')
        print('练习22.2-8 设G=(V,E)是一个连通的无向图。请给出一个O(V+E)时间的算法，以计算图G中的一条路径',
            '对于E中的每一条边,该路径都恰好在每一个方向上遍历一次')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_3:
    '''
    chpater22.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.3 note

        Example
        ====
        ```python
        Chapter22_3().note()
        ```
        '''
        print('chapter22.3 note as follow')
        print('22.3 深度优先搜索')
        # !深度搜索算法遵循的搜索策略是尽可能\"深\"地搜索一个图
        print('这种搜索算法遵循的搜索策略是尽可能\"深\"地搜索一个图')
        print('在深度优先搜索中，对于最新发现的顶点，如果还有以此为起点而未探测到的边，就沿此边继续探测下去',
            '当顶点v的所有边都已经被探寻过后，搜索将回溯到发线顶点v有起始点的那些边')
        print('这一过程一直进行到已发现从源顶点可达的所有顶点时为止')
        print('如果还存在未被发现的顶点，则选择其中一个作为源顶点，并重复以上过程')
        print('整个过程反复进行，直到所有的顶点都被发现时为止')
        print('与广度优先搜索类似，在深度优先搜索中，每当扫描已经发现顶点u的邻接表，',
            '从而发现新顶点v时,就将置v的先辈域pi[v]为u')
        print('与广度优先搜索不同的是，其先辈子图形成一棵树，深度优先搜索产生的先辈子图可以由几棵树组成',
            '因为搜索可能由多个源顶点开始重复进行。因此,在深度优先搜索中,',
            '先辈子图的定义也和广度优先搜索中稍有不同')
        print('深度优先搜索的先辈子图形成了一个由数棵深度优先树所组成的深度优先森林。Epi中边称为树边')
        print('与广度优先搜索类似，在深度优先搜索过程中，也通过对顶点进行着色来表示顶点的状态。')
        print('开始时，每个顶点均为白色，搜索中被发现时即置为灰色，结束时又被置为黑色(既当其邻接表被完全检索之后)')
        print('这一技巧可以保证每一个顶点在搜索结束时，只存在于一棵深度优先树中，因此，这些树是不相交的')
        print('除了创建一个深度优先森林外，深度优先搜索同时为每个顶点加盖时间戳。')
        print('每个顶点v由两个时间戳：当顶点v第一次被发现(并置成灰色)时，记录下第一个时间戳d[v]')
        print('每当结束检查v的邻接表(并置v为黑色)时,记录下第二个时间戳f[v]')
        print('许多图的算法中都用到了时间戳，它们对推算深度优先搜索的进行情况有很大的帮助')
        print('')
        print('')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_4:
    '''
    chpater22.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.4 note

        Example
        ====
        ```python
        Chapter22_4().note()
        ```
        '''
        print('chapter22.4 note as follow')
        print('22.4 拓扑排序')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py


class Chapter22_5:
    '''
    chpater22.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.5 note

        Example
        ====
        ```python
        Chapter22_5().note()
        ```
        '''
        print('chapter22.5 note as follow')
        print('22.5 强连通分支')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

chapter22_1 = Chapter22_1()
chapter22_2 = Chapter22_2()
chapter22_3 = Chapter22_3()
chapter22_4 = Chapter22_4()
chapter22_5 = Chapter22_5()

def printchapter22note():
    '''
    print chapter22 note.
    '''
    print('Run main : single chapter twenty-two!')  
    chapter22_1.note()
    chapter22_2.note()
    chapter22_3.note()
    chapter22_4.note()
    chapter22_5.note()

# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
if __name__ == '__main__':  
    printchapter22note()
else:
    pass
