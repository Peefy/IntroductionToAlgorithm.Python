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
    import graph as _g
    import shortestpath as _sp
else:
    from . import graph as _g
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
        print('关键路径的权值是完成所有工作所需时间的下限')
        print('练习24.2-1 如下')
        _sp.test_dag_shortest_path()
        print('练习24.2-2 略')
        print('练习24.2-3 略')
        print('练习24.2-4 给出一个高效算法统计有向无回路图中的全部路径数')
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
        print('Dijkstra算法')
        print('Dijkstra算法解决了有向图G=(V,E)上带权的单源最短路径问题,但要求所有边的权值非负',
            ',假定对每条边(u,v)∈E,有w(u,v)>=0')
        print('一个实现的很好的Dijkstra算法比Bellman-Ford算法的运行时间要低')
        print('Dijkstra算法中设置了一顶点集合S,从源点s到集合中的顶点的最终最短路径的权值均确定')
        print('算法反复选择具有最短路径估计的顶点u∈V-S,并将u加入到S中,对u的所有出边进行松弛')
        print('在下面的算法实现中,用到了顶点的最小优先队列Q,排序关键字为顶点的d值')
        # !Dijkstra算法总是在V-S中选择“最轻”或“最近”的顶点插入集合S中,使用了贪心策略
        print('Dijkstra算法总是在V-S中选择“最轻”或“最近”的顶点插入集合S中,使用了贪心策略')
        print('定理24.6 Dijkstra算法的正确性 已知一带权有向图G=(V,E),',
            '其加权函数w的值为非负,源点为s',
            '对该图运行Dijkstra算法,则在算法终止时,对所有u∈V有d[u]=d(s,u)')
        print('推论24.7 已知一加权函数非负且源点为s的带权有向图G=(V,E),若在该图上运行Dijstra算法,',
            '则在算法终止时,前趋子图Gpi是以s为根的最短路径树')
        print('Dijkstra算法的运行时间依赖于最小优先队列的具体实现')
        print(' 利用从1至|V|编好号的顶点，简单地将d[v]存入一个数组的第v项')
        print(' 每一个INSERT和DECREASE-KEY的操作都是O(1)的时间,而每一个EXTRACT-MIN操作为O(V)时间')
        print(' 总计的运行时间为O(V^2+E)=O(V^2)')
        print('特别地,如果是稀疏图的情况,有E=o(V^2/lgV),在这种情况下,',
            '利用二叉最小堆来实现最小优先队列是很有用的')
        print(' 总计的运行时间为O((V+E)lgV)')
        print('从历史的角度看,在Dijstra算法中,DECRESE-KEY的调用比EXTRACT-MIN的调用一般要多的多')
        print('所以任何能够在不增加EXTRACT-MIN操作的平摊时间的同时')
        print('从渐进意义上来说,都能获得比二叉堆更快的实现(比如斐波那契堆)')
        print('Dijkstra算法和广度优先搜索算法以及计算最小生成树的Prim算法都有类似之处')
        print('和广度优先算法的相似性在于,前者的集合S相当于后者的黑色顶点集合')
        print('练习24.3-1 ')
        _sp.test_dijstra()
        print('练习24.3-2 给出一含有负权边的有向图的简单实例,说明Dijkstra算法对其会产生错误的结果')
        print('练习24.3-3 略')
        print('练习24.3-4 已知一有向图G=(V,E),其每条边(u,v)∈E均对应有一个实数值r(u,v)',
            '表示从顶点u到顶点v之间的通信线路的可靠性,取值范围为0<=r(u,v)<=1',
            '定义r(u,v)为从u到v的线路不中断的概率,并假定这些概率是互相独立的')
        print('练习24.3-5 无权有向图G‘运行广度优先搜索,V中顶点被标记成黑色的顺序与DIJKSTRA算法运行于G上时,',
            '从优先队列中删除V中顶点的顺序相同')
        print('练习24.3-6 设G=(V,E)为带权有向图,权函数w：E->{0,1,2...,W},其中W为某非负整数。',
            '修改Dijkstra算法,以使其计算从指定源点s的最短路径所需的运行时间为O(WV+E)(在最小堆算法处加速)')
        print('练习24.3-7 略')
        print('练习24.3-8 假定有一个带权有向图G=(V,E),从源点s出发的边可能有负边,',
            '所有其他的边的权都非负,而且不存在负权回路,论证在这样的图中,Dijkstra算法可以正确地从s找到最短路径')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_4:
    '''
    chpater24.4 note and function
    '''
    def solve_24_4_1(self):
        '''
        求解练习24.4-1
        '''
        g = _g.Graph()
        g.clear()
        vertexs = ['0', '1', '2', '3', '4', '5', '6']
        g.veterxs = vertexs
        g.addedgewithweight('0', '1', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '2', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '3', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '4', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '5', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '6', 0, _g.DIRECTION_TO)
        g.addedgewithweight('1', '2', 1, _g.DIRECTION_FROM)
        g.addedgewithweight('1', '4', -4, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '3', 2, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '5', 7, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '6', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('3', '6', 10, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '2', 2, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '1', -1, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '4', 3, _g.DIRECTION_FROM)
        g.addedgewithweight('6', '3', -8, _g.DIRECTION_FROM)
        print(_sp.bellman_ford(g, vertexs[0]))
        del g
    
    def solve_24_4_2(self):
        '''
        求解练习24.4-2
        '''
        g = _g.Graph()
        g.clear()
        vertexs = ['0', '1', '2', '3', '4', '5']
        g.veterxs = vertexs
        g.addedgewithweight('0', '1', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '2', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '3', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '4', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '5', 0, _g.DIRECTION_TO)
        g.addedgewithweight('1', '2', 4, _g.DIRECTION_FROM)
        g.addedgewithweight('1', '5', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '4', -6, _g.DIRECTION_FROM)
        g.addedgewithweight('3', '2', 1, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '1', 3, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '3', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '5', 10, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '3', -4, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '4', -8, _g.DIRECTION_FROM)
        print(_sp.bellman_ford(g, vertexs[0]))

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
        print('24.4 差分约束与最短路径')
        print('一般的线性规划问题：要对一组线性不等式定义的线性函数进行优化')
        print('简化为寻找单源最短路径的线性规划的一种特殊情形')
        print('由此引出的单源最短路径问题可以运用Bellman-Ford算法来解决，进而解决原线性规划问题')
        print('线性规划')
        print('  一般的线性规划问题中，给定一个m*n的矩阵A,一个m维向量b和一个n维向量c',
            '希望找出由n个元素组成的向量x,在由Ax<=b所给出的m个约束条件下,使目标函数最大')
        print('单纯形法')
        print('  并不总是能在输入规模的多项式时间内运行；',
            '但是还有其他一些线性规划算法是可以以多项式时间运行的')
        print('有时并不关心目标函数，仅仅是希望找出一个可行解,',
            '即一个满足Ax<=b的向量x,或是确定不存在的可行解')
        print('差分约束系统')
        print('  在一个差分约束系统中，线性规划矩阵A的每一行包含一个1和一个-1',
            'A的所有其他元素都为0.因此，由Ax<=b给出的约束条件是m个差分约束集合')
        print('引理24.8 设x=(x1,x2,...,xn)是一个差分约束系统Ax<=b的一个解,d为任意常数',
            '则x+d=(x1+d,x2+d,..,xn+d)也是该系统Ax<=b的解')
        print('  差分约束系统出现在很多不同的应用领域中')
        print('约束图')
        print('  用图形理论观点来解释差分约束系统是很有益的。',
            '在一理想的差分约束系统Ax<=b,m*n的线性规划矩阵A可被看作是n顶点,m条边的图的关联矩阵的转置')
        print('  对于i=1,2,...,n图中每一个顶点vi对应着n个未知量的一个xi.',
            '图中的每个有向边对应着关于两个未知量的m个不等式的其中一个')
        print('更形式地,给定一个差分约束系统Ax<=b，相应的约束图是一个带权有向图G=(V,E),',
            '其中V={v0,v1,...,vn}')
        print('定理24.9 给定一差分系统Ax<=b，设G=(V,E)为其相应的约束图',
            '如果G不包含负权回路，那么x=(d(v0,v1),d(v0,v2),d(v0,v3),...,d(v0,vn))')
        print('是此系统的一可行解。如果G包含负权回路,那么此系统不存在可行解')
        print('差分约束问题的求解')
        print('  由定理24.9知可以采用Bellman-Ford算法对差分约束系统求解')
        print('  在约束图中，从源点v0到其他所有其他顶点均存在边，因此约束图中任何负权回路均从v0可达',
            '如果Bellman-Ford算法返回TRUE，则最短路径给出了此系统的一个可行解',
            '如果Bellman-Ford算法返回FALSE，则差分约束系统无可行解')
        print('关于n个未知量的m个约束条件的一个差分约束系统产生出一个具有n+1顶点和n+m条边的图',
            '因此采用Bell-Ford算法,可以在O((n+1)(n+m)))时间内将系统解决')
        print('可以对算法进行修改，可以使其运行时间变为O(nm),即使m远小于n')
        print('练习24.4-1 对下列差分约束系统找出其可行解,或者说明不存在可行解',
            '由差分约束不等式写出有向带权图，调用Bellman-Ford求解即可',
            '不等式左边x1-x2表示由结点2指向结点1，不等式右边表示边的权')
        self.solve_24_4_1()
        print('练习24.4-2 对下列差分约束系统找出其可行解，或者说明不存在可行解')
        self.solve_24_4_2()
        print('练习24.4-3 在约束图中，从新顶点v0出发的最短路径的权是否可以为正数')
        print('练习24.4-4 试用线性规划方法来表述单对顶点最短路径问题')
        print('练习24.4-5 试说明如何对Bellman-Ford算法稍作修改，',
            '使其在解关于n个未知量的m个不等式所定义的差分约束系统时，运行时间为O(mn)')
        print('练习24.4-6 假定除了差分约束外，还需要处理相等约束',
            '试说明Bellman-Ford算法如何作适当修改,以解决这个约束系统的变形')
        print('练习24.4-7 试说明如何不用附加顶点v0而对约束图运行类Bellman-Ford算法,从而求得差分约束系统的解')
        print('练习24.4-8 设Ax<=b是关于n个未知量的m个约束条件的差分约束系统',
            '证明对其相应的约束图运行Bellman-Ford算法,可以求得满足Ax<=b,并且对所有的xi,有xi<=0')
        print('练习24.4-9 证明Bellman-Ford算法在差分约束系统Ax<=b的约束图上运行时',
            '使(max{xi}-min{xi})取得满足Ax<=b的最小值')
        print('练习24.4-10 假设线性规划Ax<=b中，矩阵A的每一行对应于差分约束条件,即形如xi<=bk或者-xi<=bk的单变量的约束条件')
        print('练习24.4-11 对所有b的元素均为实数，且所有未知量xi必须是整数的情形,写出一个有效算法,以求得差分的约束系统Ax<=b的解')
        print('练习24.4-12 对所有b的元素均为实数且部分(并不一定是全部)未知量xi必须是整数的情形,',
            '写出一个有效算法,以求得差分的约束系统Ax<=b')
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
        print('论证的几种正确性均依赖于三角不等式、上界性质、',
            '无路径性质、收敛性质、路径松弛性质和前趋子图性质')
        print('引理24.10(三角不等式) 设G=(V,E)为一带权有向图，其权函数w:E->R,源点为s',
            '那么对于所有边(u,v)∈E,有d(s,v)<=d(s,u)+w(u,v)')
        print('对最短路径估计的松弛的效果')
        print('引理24.11(上界性质)设G=(V,E)为有向加权图,其加权函数为w：E->R',
            '设s∈V为源点,INITIALIZE-SINGLE-SOURCE(G,s)对图进行了初始化',
            '那么，对于所有v∈V有d[v]>=d(s,v),而且这个不变式对图G中边的任意系列松弛操作都保持不变',
            '更进一步说,一旦d[v]到达下界d(s,v)将不再改变')
        print('推论24.12(无路径性质)假定在给定的带权有向图G=(V,E)中,权函数为w:E->R',
            '从一源点s∈V到一给定顶点v∈V不存在路径。',
            '那么在INITIALIZE-SINGLE-SOURCE(G,s)对图初始化以后,有d[v]=d(s,v)=∞',
            '在对于G的边进行任意序列的松弛操作后,这个等式作为循环不变式仍然保持')
        print('引理24.13 设G=(V,E)为一个带权有向图,其权函数w:E->R,且(u,v)∈E。',
            '那么,通过执行RELAX(u,v,w)松弛边(u,v)后.有d[v]<=d[u]+w(u,v)')
        print('引理24.14(收敛性质) 设G=(V,E)为一个带权有向图,其权函数为w:E->R,s∈V为一个源点',
            '对某些顶点u,v∈V，设s~u->v为图G中的最短路径。假定G通过调用INITIALIZE-SINGLE-SOURCE(G,s)进行初始化',
            '然后在图G的边上执行了包括调用RELAX(u,v,w)在内的一系列松弛步骤',
            '如果在调用之前d[u]=d(s,u),那么在调用之后的任意时间d[v]=d(s,v)')
        print('引理24.15(路径松弛性质)是G=(V,E)为一带权有向图，权函数w:E->R.s∈V为源点',
            '考虑任意从s=v0到vk的最短路径p=<v0,v1,...,vk>,如果G通过INITIALIZE-SINGLE-SOURCE(G,s)',
            '然后按顺序进行了一系列的松弛步骤,包括松弛边(v0,v1),(v1,v2),...,(vk-1,vk)',
            '那么，经过这些松弛后以及在以后的任意时刻，都有d[vk]=d(s,vk).',
            '无论其他边是否发生松弛(包括与p的边交错地进行的松弛),这一性质都始终保持')
        print('引理24.16 设G=(V,E)为一带权有向图,其权值函数为w:E->R,s∈V为一个源点',
            '并假定G不含从s可达的负权回路，那么，在图INITIALIZE-SINGLE-SOURCE(G,s)初始化后',
            '前趋子图Gpi就构成以s为根的有根树,在对G边任意序列的松弛操作下仍然像不变式一样保持这个性质')
        print('引理24.17(前趋子图性质)设G=(V,E)为一带权有向图,其权函数w:E->R,s∈V为一个源点',
            '而且假定G不含s可达的负权回路.设调用了INITIALIZE-SINGLE-SOURCE(G,s)',
            '然后在G的边上执行了一系列的松弛操作,得到对所有v∈V有d[v]=d(s,v)',
            '因此,前趋子图Gpi是一个以s为根的最短路径树')
        print('练习24.5-1 对图24-2，除图中已画出的两棵树以外,另外再画出两棵图中所示有向图的最短路径树')
        print('练习24.5-2 举出一个带权有向图G=(V,E)的实例，其加权函数为w：E->R,且源点为s,',
            '要求G满足下列性质：对每条边(u,v)∈E，存在包含(u,v)且以s为根的最短路径树',
            '同时存在另一棵以s为根，但不包含(u,v)的最短路径树')
        print('练习24.5-3 略')
        print('练习24.5-4 设G=(V,E)是带权有向图,源点为s,并设G由过程INITIALIZE=SINGLE-SOURCE(G,s)进行了初始化',
            '证明如果经过一系列松弛操作,pi[x]的值被置为非None,则G中包含一个负权回路')
        print('练习24.5-5 设G=(V,E)为带权有向图且不含负权边，设s∈V为源点若v∈V-{s}为从s可达的顶点',
            '则pi[v]是从源s到v的某最短路径中顶点v的前趋,否则pi[v]=None',
            '举出这样的一个图G和给pi赋值的一个例子,说明可以在Gpi中产生回路')
        print('练习24.5-6 设G=(V,E)为带权有向图,其权值函数为w:E->R,且图中不包含负权回路',
            '设s∈V为源点,且G由INITIALIZE-SINGLE-SOURCE(G,s)进行了初始化',
            '证明对每一顶点v∈Vpi,Gpi中存在一条从s到v的通路,且经过任意序列的松弛操作后，这一性质仍然保持')
        print('练习24.5-7 设G=(V,E)为带权有向图且不包含负权回路.设s∈V为源点且G由INITIALIZE-SINGLE-SOURCE(G,s)进行了初始化',
            '证明存在|V|-1步的松弛序列,使得对所有v∈V,d[v]=d(s,v)')
        print('练习24.5-8 设G为任意带权有向图,且存在一源点s可达负权回路。',
            '证明对G的边总可以构造一个无限的松弛序列，使得每个松弛步骤都能对最短路径估计进行修改')
        print('思考题24-1 对Bellman-Ford算法的Yen氏改进')
        print(' 假设对Bellman-Ford算法每一趟中边的松弛顺序作如下安排,在第一趟执行之前',
            '把一任意线性序列v1,v2,...,v|v|赋值给输入图G=(V,E)的各点')
        print(' a)证明对拓扑序列<v1,v2,...,v|v|>,Gf是无回路图；对拓扑序列<v(|v|),v(|v|-1),..,v(1)>')
        print('思考题24-2 嵌套框')
        print(' 如果存在{1,2,...,d}上的某一排列pi,满足xn(1)<y1,xpi(2)<=y2,...,xpi(d)<yd嵌入另一个d维框(y1,y2,..,yd)中')
        print(' a)证明嵌套关系具有传递性')
        print(' b)描述一个有效算法以确定某d维框是否嵌套于另一d维框中')
        print(' c)假定给出一个由n个d维框组成的集合{B1,B2,...,Bn},写出有效算法以找出满足条件Bij嵌入',
            'Bij+1,j=1,2,..,k-1的最长嵌套框序列<Bi1,Bi2,...,Bik>','用变量n和d来描述所给出的算法的运行时间')
        print('思考题24-3 套汇问题')
        print(' 套汇是指利用货币兑率的差异，把一个单位的某种货币转换为大于一个单位的同种货币的方法')
        print('思考题24-4 关于单源最短路径的Gabow定标算法')
        print(' 定标算法对问题进行求解,开始时仅考虑每个相位输入值(例如边的权)的最高位，',
            '接着通过查看最高两位对初始答案进行细微调整,这样逐步查看越来越多的高位信息')
        print('思考题24-5 Karp最小平均权值回路算法')
        print(' 某边回路包含的所有边的平均权值')
        print('思考题24-6 双调最短路径')
        print(' 如果一个序列首先单调递增，然后再单调递减，',
            '或者能够通过循环移位来单调递增再单调递减,这样的序列就是双调的')
        print('')
        print('')
        print('')
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
