# coding:utf-8
# usr/bin/python3
# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
'''

Class Chapter23_1

Class Chapter23_2


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
    import mst as _mst
else: 
    from . import mst as _mst

class Chapter23_1:
    '''
    chpater23.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.1 note

        Example
        ====
        ```python
        Chapter23_1().note()
        ```
        '''
        print('chapter23.1 note as follow')  
        print('设计电子线路时，如果要使n个引脚互相连通,可以使用n-1条连接线',
            '每条连接线连接两个引脚。在各种链接方案中，通常希望找出连接线最少的接法')
        print('可以把这一接线问题模型化为一个无向连通图G=(V,E)',
            '其中V是引脚集合，E是每对引脚之间可能互联的集合')
        print('对图中每一条边(u,v)∈E,都有一个权值w(u,v)表示连接u和v的代价(需要的接线数目)')
        print('希望找出一个无回路的子集T∈E,它连接了所有的顶点，且其权值之和w(T)=∑w(u,v)最小')
        print('因为T无回路且连接了所有的顶点,所以它必然是一棵树，称为生成树')
        print('因为由最小生成树可以\"生成\"图G')
        print('把确定树T的问题称为最小生成树问题')
        print('最小生成树问题的两种算法：Kruskal算法和Prim算法')
        print('这两种算法中都使用普通的二叉堆，都很容易达到O(ElgV)的运行时间')
        print('通过采用斐波那契堆，Prim算法的运行时间可以减小到O(E+VlgV),',
            '如果|V|远小于|E|的话,这将是对该算法的较大改进')
        print('这两个算法都是贪心算法，在算法的每一步中，都必须在几种可能性中选择一种')
        print('贪心策略的思想是选择当时最佳的可能性，一般来说，这种策略不一定能保证找到全局最优解')
        print('然而，最小生成树问题来说,却可以证明某些贪心策略的确可以获得具有最小权值的生成树')
        print('23.1 最小生成树的形成')
        print('假设已知一个无向连通图G=(V,E),其权值函数为w')
        print('目的是找到图G的一棵最小生成树')
        print('通用最小生成树算法')
        print('在每一个步骤中都形成最小生成树的一条边,算法维护一个边的集合A,保持以下的循环不变式:')
        print(' 在每一次循环迭代之前，A是某个最小生成树的一个子集')
        print(' 在算法的每一步中，确定一条边(u,v)，使得将它加入集合A后，仍然不违反之歌循环不变式;',
            '亦即，A∪{(u,v)}仍然是某一个最小生成树的子集')
        print(' 称这样的边为A的安全边(safe edge),因为可以安全地把它添加到A中,而不会破坏上述的循环不变式')
        print('在算法的执行过程中，集合A始终是无回路的，否则包含A的最小生成树将包含一个环')
        print('无向图G=(V, E)的一个割(S, V-S)是对V的一个划分.当一条边(u,v)∈E的一个端点属于S，而另一个端点属于V-S',
            '则称边(u,v)通过割(S,V-S).如果一个边的集合A中没有边通过某一割','则说该割不妨害边集A')
        print('如果某条边的权值是通过一个割的所有边中最小的,则称该边为通过这个的割的一条轻边(light edge)')
        print('GENERIC-MST')
        print('  A = []')
        print('  while A does not form a spanning tree')
        print('    do find an edge (u,v) that is safe for A (保证不形成回路)')
        print('       A <- A ∪ {(u, v)}')
        print('  return A')
        print('')
        print('识别安全边的一条规则：')
        print('定理23.1 设图G=(V,E)是一个无向连通图，并且在E上定义了一个具有实数值的加权函数w.',
            '设A是E的一个子集，它包含于G的某个最小生成树中.',
            '设割(S,V-S)是G的任意一个不妨害A的割,且边(u,v)是通过集合A来说是安全的')
        print('推论23.2 设G=(V,E)是一个无向连通图,并且在E上定义了相应了实数值加权函数w',
            '设A是E的子集，且包含于G的某一最小生成树中。设C=(Vc,Ec)为森林GA=(V,A)的一个连通分支(树)',
            '如果边(u,v)是连接C和GA中其他某联通分支的一条轻边,则(u,v)对集合A来说是安全的')
        print('证明:因为割(Vc,V-Vc)不妨害A，(u,v)是该割的一条轻边。因此(u,v)对A来说是安全的')
        print('练习23.1-1 设(u,v)是图G中的最小权边.证明:(u,v)属于G的某一棵最小生成树')
        print('练习23.1-2 略')
        print('练习23.1-3 证明：如果一条边(u,v)被包含在某一最小生成树中,那么它就是通过图的某个割的轻边')
        print('练习23.1-4 因为这条边虽然是轻边，但是连接后产生不安全的回路')
        print('练习23.1-5 设e是图G=(V,E)的某个回路上一条最大权边.证明：存在着G\'=(V,E-{e})的一棵最小生成树,',
            '它也是G的最小生成树。亦即，存在着G的不包含e的最小生成树')
        print('练习23.1-6 证明：一个图有唯一的最小生成树,如果对于该图的每一个割,都存在着通过该割的唯一一条轻边',
            '但是其逆命题不成立')
        print('练习23.1-7 论证：如果图中所有边的权值都是正的，那么，任何连接所有顶点、',
            '且有着最小总权值的边的子集必为一棵树')
        print('练习23.1-8 设T是图G的一棵最小生成树，L是T中各边权值的一个已排序的列表',
            '证明：对于G的任何其他最小生成树T\'，L也是T\'中各边权值的一个已排序的列表')
        print('练习23.1-9 设T是图G=(V,E)的一棵最小生成树,V\'是V的一个子集。设T\'为T的一个基于V\'的子图',
            'G\'为G的一个基于V\'的子图。证明:如果T\'是连通的,则T\'是G\'的一棵最小生成树')
        print('练习23.1-10 给定一个图G和一棵最小生成树T,假定减小了T中某一边的权值。',
            '证明：T仍然是G的一棵最小生成树。更形式地,设T是G的一棵最小生成树',
            '其各边的权值由权值函数w给出.')
        print(' 证明：T是G的一棵最小生成树，其各边的权值由w\'给出')
        print('练习23.1-11 给定一个图G和一棵最小生成树T，假定减小了不在T中的某条边的权值',
            '请给出一个算法,来寻找经过修改的图中的最小生成树')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

class Chapter23_2:
    '''
    chpater23.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.2 note

        Example
        ====
        ```python
        Chapter23_2().note()
        ```
        '''
        print('chapter23.2 note as follow')
        print('23.2 Kruskai算法和Prim算法')
        print('本节所介绍的两种最小生成树算法是对上一节介绍的通用算法的细化')
        print('均采用了一个特定的规则来确定GENERIC-MST算法所描述的安全边')
        print(' 在Kruskal算法中,集合A是一个森林,加入集合A中的安全边总是图中连接两个不同连通分支的最小权边')
        print(' 在Prim算法中,集合A仅形成单棵树,',
            '添加入集合A的安全边总是连接树与一个不在树中的顶点的最小权边')
        print('Kruskal算法')
        print(' 该算法找出森林中连接任意两棵树的所有边中,具有最小权值的边(u,v)作为安全边',
            '并把它添加到正在生长的森林中')
        print(' 设C1和C2表示边(u,v)连接的两棵树,因为(u,v)必是连接C1和其他某棵树的一条轻边',
            '所以由推论23.2可知,(u,v)对C1来说是安全边。Kruskal算法同时也是一种贪心算法',
            '因为在算法的每一步中,添加到森林中的边的权值都是尽可能小的')
        _mst.test_mst_kruskal()
        print(' Kruskal算法在图G=(V,E)上的运行时间取决于不相交集合数据结构是如何实现的')
        print(' 使用*按秩结合*和*路径压缩*的启发式方法实现不相交集合森林，从渐进意义上来说是最快的方法')
        print(' 综上所述：算法总的运行时间所需时间为O(ElgE),由于E<V^2',
            '因而lgE=O(lgV),于是也可以将Kruskal算法的运行时间重新表述为O(ElgV)')
        print('Prism算法')
        print(' 如Kruskal算法一样,Prism算法也是通用最小生成树算法的特例')
        print(' Prism算法的执行非常类似于寻找图的最短路径的Dijkstra算法')
        print(' Prism算法的特点是集合A中边总形成单棵树')
        print(' 树从任意根顶点r开始形成,并逐渐形成,直至该树覆盖了V中所有的顶点')
        print(' 在每一步，一条连接了树A与GA=(V,A)中某孤立顶点的轻边被加入到树A中')
        print(' 由推论23.2可知，该规则仅加入对A安全的边，因此当算法终止时,',
            'A中的边就形成了一棵最小生成树')
        print(' 因为每次添加到树中的边都是使树的权尽可能的小的边.因此策略也是贪心的')
        print('有效实现Prism算法的关键是设法较容易地选择一条新的边,将其添加到由A的边所形成的树中',
            '算法的输入是连通图G和待生成的最小生成树根r')
        print('在算法的执行过程中,不在树中的所有顶点都放在一个基于key域的最小优先级队列Q中')
        print('对每个顶点v来说,key[v]是所有将v与树中某一顶点相连的边中的最小权值;')
        print('按据约定,如果不存在这样的边,则key[v]=∞,pi[v]=None')
        print('当算法终止时,最小优先队列Q是空的,而G的最小生成树A则满足：')
        print(' A={(v,pi[v]):v∈V-{r}}')
        print('Prism算法的性能,取决于优先队列Q是如何实现的,如果用二叉最小堆实现,其运行时间为O(V)',
            '由于EXTRACT-MIN操作需要O(lgV)时间,所以对EXTRACT-MIN的全部调用所占用的之间为O(VlgV)')
        print('通过使用斐波那契堆,Prism的算法渐进运行时间可得到进一步改善,',
            '可在O(lgV)的平摊时间内完成EXTRACT-MIN操作,在O(1)的平摊时间里完成DECRESE-KEY操作')
        print('因此,如果使用斐波那契堆来实现最小优先队列Q,Prism算法的运行时间可以改进为O(E+VlgV)')
        print('练习23.2-1 根据对边进行排序不同，即使对同一输入图,Kruskal算法也可能得出不同的生成树',
            '证明对G的每一棵最小生成树T,Kruskal算法中都存在一种方法来对边进行排序,使得算法返回的最小生成树为T')
        print('练习23.2-2 假定图G=(V,E)用邻接矩阵表示,在这种条件下,给出Prism算法的运行时间为O(V^2)的实现')
        print('练习23.2-3 稀疏图G=(V,E),|E|=Θ(V),稠密图|E|=Θ(V^2)')
        print('练习23.2-4 因为已经知道了权值的上限，采用计数排序进行权重排序加速')
        print('练习23.2-5 使用斐波那契堆进行加速排序')
        print('练习23.2-6 使用桶排序进行加速排序')
        print('练习23.2-7 假设某个图G有一棵已经计算出来的最小生成树。',
            '如果一个新的顶点以及关联的边被加入到了G中,该最小生成树可以多块的时间内被更新')
        print('练习23.2-8 分治算法计算最小生成树，给定一个图G=(V,E),将顶点集合V划分成两个集合V1和V2',
            '使得|V1|和|V2|至多差1.设E1为一个边集,其中的边都与V1中的顶点关联',
            'E2为另一个边集,其中的边都与V2中的顶点关联.在两个子图G1=(V1,E1)和G2=(V2,E2)上,',
            '分别递归地解决最小生成树问题.最后，从E中选择一条通过割集(V1,V2)的最小权边,',
            '并利用该边,将所得的两棵最小生成树合并成一棵完整的生成树')
        print('思考题23-1 次最优的最小生成树')
        print(' a)证明最小生成树是唯一的,但次最优最小生成树未必一定是唯一的')
        print(' b)设T是G的一棵最小生成树,证明存在边(u,v)∈T和(x,y)∉T,',
            '使得T-{(u,v)}∪{(x,y)}是G的一棵次最优最小生成树')
        print(' c)设T是G的一棵生成树,且对任意两个顶点u,v∈V,设max[u,v]是T中u和v',
            '之间唯一通路上的具有最大权值的边.请给出一个运行时间为O(V^2)的算法,',
            '在给定T和所有顶点u,v∈V以后,可以计算出max[u,v]')
        print(' 最小生成树也是无向无环连通图')
        print('思考题23-2 稀疏图的最小生成树')
        print(' ')
        print('思考题23-3 瓶颈生成树')
        print(' ')
        print('思考题23-4 其他最小生成树算法')
        print(' ')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

chapter23_1 = Chapter23_1()
chapter23_2 = Chapter23_2()

def printchapter23note():
    '''
    print chapter23 note.
    '''
    print('Run main : single chapter twenty-three!')  
    chapter23_1.note()
    chapter23_2.note()

# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
if __name__ == '__main__':  
    printchapter23note()
else:
    pass
