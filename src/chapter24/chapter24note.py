# coding:utf-8
# usr/bin/python3
# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
'''

Class Chapter24_1

Class Chapter24_2

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
    import shortestpath as _sp
else:
    from . import shortestpath as _sp

class Chapter24_1:
    '''
    chpater24.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.1 note

        Example
        ====
        ```python
        Chapter24_1().note()
        ```
        '''
        print('chapter24.1 note as follow')  
        print('第24章 单源最短路径')
        print('一种求最短路径的方式就是枚举出所有从芝加哥到波士顿的路线,',
            '并对每条路线的长度求和,然后选择最短的一条')
        print('在最短路径问题中,给出的是一个带权有向图G=(V,E),加权函数w:E->R为从边到实型权值的映射')
        print('路径p=<v0,v1,...,vk>的权是指其组成边的所有权值之和')    
        print('边的权值还可以被解释为其他的某种度量标准,而不一定是距离')
        print('它常常被用来表示时间,费用,罚款,损失或者任何其他沿一',
            '条路径线性积累的试图将其最小化的某个量')
        # !广度优先搜索算法就是一种在无权图上执行的最短路径算法
        print('广度优先搜索算法就是一种在无权图上执行的最短路径算法,',
            '即在图的边都具有单位权值的图上的一种算法')
        print('单源最短路径的变体')
        print(' 已知图G=(V,E),希望找出从某给定源顶点s∈V到每个顶点v∈V的最短路径。')
        print('很多其他问题都可用单源问题的算法来解决,其中包括下列变体')
        print(' 1.单终点最短路径问题:找出从每个顶点v到指定终点t的最短路径')
        print(' 2.单对顶点最短路径问题:对于某给定顶点u和v,找出从u和v的一条最短路径')
        print('   如果解决了源点为u的单源问题,则这一问题也就获得解决')
        print(' 3.对于每对顶点u和v,找出从u到v的最短路径')
        print('   虽然将每个顶点作为源点,运行一次单源算法就可以解决这一问题,但通常可以更快地解决这一问题')
        print('最短路径的最优子结构')
        print('  最短路径算法通常依赖于一种性质,也就是一条两顶点间的最短路',
            '径包含路径上其他的最短路径,这种最优子结构性质是动态规划和贪心算法是否适用的一个标记')
        print('Dijkstra算法是一个贪心算法,而找出所有顶点对之间的最短路径的',
            'Floyd-Warshall算法是一个动态规划算法')
        print('引理24.1(最短路径的子路径是最短路径)对于一给定的带权有向图G=(V,E),所定义的权函数为w',
            'E->R。设p=<v1,v2,..,vk>是从v1到vk的最短路径')
        print('负权值边')
        print(' 在单源最短路径问题的某些实例中,可能存在着权值为负值的边.',
            '如果图G=(V,E)不包含从源s可达的负权回路,则对所有v∈V,最短路径的权的定义d(u,v)依然正确,',
            '即使它是一个负值也是如此.但是,如果存在一条从s可达的负权回路,那么最短路径的权的定义就不能成立')
        print('从s到该回路上的顶点之间就不存在最短路径,因为我们总是可以顺着已找出的\"最短\"路径,',
            '再穿过负权值回路而获得一条权值更小的路径.',
            '因此,如果从s到v的某路径中存在一条负权回路,就定义d(u,v)=-inf')
        print('一些最短路径算法,如Dijstra算法,假定输入图中的所有边的权值都是非负的,如公路地图的例子',
            '另一些算法,如Bellman-Ford算法,允许输入图中存在负权边,只要不存在从源点可达的负权回路')
        print('特别地,如果存在负权回路,算法还可以检测并报告这种回路的存在')
        print('一条最短路径能包含回路嘛?不能包含负权回路.也不会包含正权回路,因为从路径上移去回路后,',
            '可以产生一个具有相同源点和终点、权值更小的路径')
        print('最短路径的表示')
        print('不仅希望算出最短路径的权,而且也希望得到最短路径设置pi属性,',
            '以便使源于顶点v的前辈链表沿着从s到v的最短路径的相反方向排列')
        print('对于一给定的v.pi=None的顶点v,',
            '可以运用PRINT-PATH(G,s,v)输出从s到v的一条最短路径')
        print('不过,在最短路径算法的执行过程中,无需用pi的值来指明最短路径。',
            '正如广度优先搜索一样,是由pi值导出的前趋子图Gpi=(Vpi,Epi).这里,',
            '定义顶点集Vpi为G中所有具有非空前趋的顶点集合,再加上源点s')
        # !最短路径并不一定是唯一的,最短路径树亦是如此
        print('最短路径并不一定是唯一的,最短路径树亦是如此')
        print('松弛技术')
        print(' 本章的算法用到了松弛(relaxation)技术.',
            '对每个顶点v∈V,都设置一个属性d[v],用来描述从源点s到v',
            '的最短路径上权值的上界,称为最短路径估计')
        print('在松弛一条边(u,v)的过程中,要测试是否可以通过u,对找到的到v的最短路径进行改进',
            '如果可以改进的话,则更新d[v]和pi[v].一次松弛操作可以减小最短路径估计的值d[v]',
            '并更新v的前趋pi[v]')
        print('本章每个算法都会调用INITIALIZE-SINGLE-SOURCE,然后重复对边进行松弛的过程',
            '另外,松弛是改变最短路径和前趋的唯一方式')
        print('本章中的算法之间的区别在于对每条边进行松弛操作的次数',
            '以及对边执行松弛操作的次序有所不同')
        print('在Dijkstra算法以及关于有向无回路图的最短路径算法中,对每条边执行一次松弛操作')
        print('在Bellman-Ford算法中,对每条边要执行多次松弛操作')
        print('最短路径以及松弛的性质(隐含地假设了图是调用INITIALIZE-SINGLE-SOURCE(G,s)进行初始化的,'
            '且最短路径估计和前趋子图唯一的变化途径就是一系列的松弛步骤')
        print('1.三角不等式(引理24.10)')
        print('  对任意边(u,v)∈E,有d(s,v)<=d(s,u)+w(u,v)')
        print('2.上界性质(引理24.11)')
        print('  对任意顶点v∈V,有d[v]>=d(s,v),而且一旦d[v]达到d(s,v)值就不再改变')
        print('3.无路径性质(推论24.12)')
        print('  如果从s到v不存在路径,则总是有d[v]=d(s,v)=inf')
        print('4.收敛性质(引理24.14)')
        print('  如果s-u->v是图G某u,v∈V的最短路径,而且在松弛边(u,v)',
            '之前的任何时间d[u]=d(s,u),则在操作之后总有d[v]=d(s,v)')
        print('路径松弛性质(引理24.15)')
        print('这个性质的保持并不受其他松弛操作的影响,',
            '即使它们与p的边上的松弛操作混合在一起也是一样的')
        print('前趋子图性质(引理24.17)')
        print('  一旦对于所有v∈V,d[v]=d(s,v),前趋子图就是一个以s为根的最短路径树')
        print('Bellman-Ford算法,该算法用来解决一般(边的权值可以为负)的单源最短路径问题')
        print('Bellman-Ford算法非常简单,可以检测是否有从源点可达的负权回路')
        print('在一个有向无环图中,在线性时间内计算出单源最短路径的算法')
        print('Dijkstra算法,它的运行时间比Bellman-Ford算法低,但要求所有边的权值为非负')
        print('使用Bellman-Ford算法来解决\"动态规划\"的一个特例')
        print('24.5节证明了上面所陈述的最短路径和松弛的性质')
        print('所有算法都假设有向图G用邻接表的形式存储,而且每条边上还存储了它的权值')
        print('当遍历每一个邻接表时,可以对每条边在O(1)时间内确定其权值')
        print('24.1 Bellman-Ford算法')
        print('Bellmax-Ford算法能在一般的情况下(存在负边权的情况)下,解决单源最短路径问题',
            '对于给定的带权有向图G=(V,E),其源点为s,加权函数为w')
        print('Bellman-Ford算法后可以返回一个布尔值,表明图中是否存在着一个从源点可达的权为负的回路')
        print('若存在这样的回路的话,算法说明该问题无解;若不存在这样的回路,算法将产生最短路径及其权值')
        print('算法运用松弛技术,对每个顶点v∈V,逐步减小源s到v的最短路径的权的估计值d[v]直至其达到的实际最短路径的权d(s,v)')
        print('算法返回布尔值TRUE,当且仅当图中不包含从源点可达的负权回路')
        print('引理24.2 设G=(V,E)为带权有向图,其源点为s,权函数为w:E->R,',
            '并且假定G中不包含从s点可达的负权回路')
        print('推论24.3 设G=(V,E)为带权有向图,源顶点为s,加权函数为w,E->R.',
            '对每一顶点v∈V,从s到v存在一条通路','当且仅当对G运行Bellman-Ford算法,算法终止时,有d[v]<∞')
        print('定理24.4(Bellman-Ford算法的正确性),设G=(V,E)为带权有向图.源点为s,权函数为w：E->R',
            '对该图运行Bellman-Ford算法.若G不包含s可达的负权回路,则算法返回TRUE',
            '对所有顶点v∈V,有d[v]=d(s,v)成立.前趋子图Gpi是以s为根的最短路径树',
            '如果G包含从s可达的负权回路,则算法返回FALSE')
        print('练习24.1-1 以顶点z作为源点,对图24-4所给出的有向图运行Bellman-Ford算法',
            '每趟操作中,按照图中的相同顺序对边进行松弛,并表示出每趟过后d与pi的值',
            '现在,将边(z,x)的权值变为4,再以s为源点运行此算法')
        _sp.test_bellman_ford()
        print('练习24.1-2 证明推论24.3')
        print('练习24.1-3 对于给定的无负权回路的带权有向图G=(V,E),设在所有u,v∈V的顶点对中,',
            'm为所有从u到为v的最短路径上边数最小值中的最大值(这里,最短路径是根据权值来说的,而不是边的数目)',
            '可以对Bellman-Ford算法做简单的修改,则可在m+1趟后终止')
        print('练习24.1-4 对Bellman-Ford算法进行比较,对任意顶点v,',
            '当从源点到v的某些路径上存在一个负权回路,则置d[v]=-∞')
        print('练习24.1-5 设G=(V,E)为一带权有向图,其权函数w:E->R。请给出一个O(VE)时间的算法',
            '对每个顶点v∈V,找出d(v)=min{d(u,v)}')
        print('练习24.1-6 假定一加权有向图G=(V,E)包含一负权回路.',
            '请给出一个能够列出此回路上的顶点的高效算法')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_2:
    '''
    chpater24.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.2 note

        Example
        ====
        ```python
        Chapter24_2().note()
        ```
        '''
        print('chapter24.2 note as follow')
        print('24.2 有向无回路图中的单源最短路径')
        print('按顶点的拓扑序列对某加权dag图(有向无回路图)G=(V,E)的边进行松弛后',
              '就可以在Θ(V+E)时间内计算出单源最短路径.在一个dag图中最短路径总是存在的',
            '因为即使图中有权值为负的边，也不可能存在负权回路')
        print('定理24.5 如果一个带权有向图G=(V,E)有源点s而且无回路',
            '则在DAG-SHORTEST-PATHS终止时,对任意顶点v∈V,有d[v]=d(s,v),',
            '且前趋子图Gpi是最短路径树')
        print('DAG-SHORTEST-PATHS算法一个有趣的应用是在PERT图分析中确定关键路径',
            '在PERT图中,边表示要完成的工作,边的权表示完成特定工作所需时间',
            '如果边(u,v)进入顶点v而边(v,x)离开顶点v,则工作(u,v)必须在工作(v,x)之前完成')
        print('此dag的一个路径表示必须按一定顺序执行工作序列.关键路径是通过dag的一条最长路径,',
            '它对应于执行一个有序的工作序列的最长时间')
        print('故那劲啊路径的权值是完成所有工作所需时间的下限')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_3:
    '''
    chpater24.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.3 note

        Example
        ====
        ```python
        Chapter24_3().note()
        ```
        '''
        print('chapter24.3 note as follow')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_4:
    '''
    chpater24.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.4 note

        Example
        ====
        ```python
        Chapter24_4().note()
        ```
        '''
        print('chapter24.4 note as follow')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_5:
    '''
    chpater24.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.5 note

        Example
        ====
        ```python
        Chapter24_5().note()
        ```
        '''
        print('chapter24.5 note as follow')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

chapter24_1 = Chapter24_1()
chapter24_2 = Chapter24_2()

def printchapter24note():
    '''
    print chapter24 note.
    '''
    print('Run main : single chapter twenty-four!')  
    chapter24_1.note()
    chapter24_2.note()

# python src/chapter24/chapter24note.py
# python3 src/chapter24/chapter24note.py
if __name__ == '__main__':  
    printchapter24note()
else:
    pass
