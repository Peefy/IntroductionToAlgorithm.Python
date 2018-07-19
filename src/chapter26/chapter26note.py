# coding:utf-8
# usr/bin/python3
# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
'''

Class Chapter26_1

Class Chapter26_2

Class Chapter26_3

Class Chapter26_4

Class Chapter26_5

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

class Chapter26_1:
    '''
    chpater26.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.1 note

        Example
        ====
        ```python
        Chapter26_1().note()
        ```
        '''
        print('chapter26.1 note as follow')  
        print('第26章 最大流')
        print('为了求从一点到另一点的最短路径，可以把公路地图模型化为有向图')
        print('可以把一个有向图理解为一个流网络,并运用它来回答有关物流方面的问题')
        print('设想某物质从产生它的源点经过一个系统,流向消耗该物质的汇点(sink)这样一种过程')
        print('源点以固定速度产生该物质,而汇点则用同样的速度消耗该物质.',
            '从直观上看,系统中任何一点的物质的流为该物质在系统中运行的速度')
        print('物质进入某顶点的速度必须等于离开该顶点的速度,流守恒性质,',
            '当物质是电流时,流守恒与基尔霍夫电流定律等价')
        print('最大流问题是关于流网络的最简单的问题')
        print('26.1 流网络')
        print('流网络的流')
        print('  流网络G=(V,E)是一个有向图,其中每条边(u,v)∈E均有一非负容量c(u,v)>=0',
            '如果(u,v)∉E,则假定c(u,v)=0。流网络中有两个特别的顶点：源点s和汇点t',
            '为了方便起见，假定每个顶点均处于从源点到汇点的某条路径上，就是说,每个顶点v∈V,存在一条路径s->v->t',
            '因此,图G为连通图,且|E|>=|V|-1')
        print('流的定义')
        print('  设G=(V,E)是一个流网络,其容量函数为c。设s为网络的源点，t为汇点。',
            'G的流是一个实值函数f:V*V->R,且满足下列三个性质：')
        print('  (1) 容量限制：对所有u,v∈V,要求f(u,v)<=c(u,v)')
        print('  (2) 反对称性：对所有u,v∈V,要求f(u,v)=-f(v,u)')
        print('  (3) 流守恒性：对所有u∈V-{s,t},要求∑f(u,v)=0')
        print('  f(u,v)称为从顶点u到顶点v的流，可以为正，为零，也可以为负。流f的值定义为|f|=∑f(s,v)',
            '即从源点出发的总流.在最大流问题中,给出一个具有源点s和汇点t的流网络G,希望找出从s到t的最大值流')
        print('容量限制只说明从一个顶点到另一个顶点的网络流不能超过设定的容量',
            '反对称性说明从顶点u到顶点v的流是其反向流求负所得.',
            '流守恒性说明从非源点或非汇点的顶点出发的总网络流为0')
        print('定义某个顶点处的总的净流量(total net flow)为离开该顶点的总的正能量,减去进入该顶点的总的正能量')
        print('流守恒性的一种解释是这样的,即进入某个非源点非汇点顶点的正网络流，必须等于离开该顶点的正网络流',
            '这个性质(即一个顶点处的总的净流量必定为0)常常被形式化地称为\"流进等于流出\"')
        print('通常，利用抵消处理，可以将两城市间的运输用一个流来表示,该流在两个顶点之间的至多一条边上是正的')
        print('给定一个实际运输的网络流f,不能重构其准确的运输路线,如果知道f(u,v)=5,',
            '如果知道f(u,v)=5,表示有5个单位从u运输到了v,或者表示从u到v运输了8个单位,v到u运输了3个单位')
        print('本章的算法将隐式地利用抵消,假设边(u,v)有流量f(u,v).在一个算法的过程中,可能对边(u,v)上的流量增加d',
            '在数学上,这一操作为f(u,v)减d；从概念上看,可以认为这d个单位是对边(u,v)上d个单位流量的抵消')
        print('具有多个源点和多个汇点的网络')
        print('  在一个最大流问题中,可以有几个源点和几个汇点,而非仅有一个源点和一个汇点',
            '比如物流公司实际可能拥有m个工厂{s1,s2,...,sm}和n个仓库{t1,t2,...,tn}',
            '这个问题不比普通的最大流问题更难')
        print('  在具有多个源点和多个汇点的网络中,确定最大流的问题可以归约为一个普通的最大流问题',
            '通过增加一个超级源点s,并且对每个i=1,2,...,m加入有向边(s,si),其容量c(s,si)=∞',
            '同时创建一个超级汇点t,并且对每个j=1,2,...,n加入有向边(tj,t),其容量c(tj,t)=∞')
        print('  单源点s对多个源点si提供了其所需要的任意大的流.同样,单汇点t对多个汇点tj消耗其所需要的任意大的流')
        print('对流的处理')
        print('  下面来看一些函数(如f),它们以流网络中的两个顶点作为自变量',
            '在本章,将使用一种隐含求和记号,其中任何一个自变量或两个自变量可以是顶点的集合',
            '它们所表示的值是对自变量所代表元素的所有可能的情形求和')
        print('  流守恒限制可以表述为对所有u∈V-{s,t},有f(u,V)=0,',
            '同时,为方便起见,在运用隐含求和记法时,省略集合的大括号.例如,在等式f(s,V-s)=f(s,V)中',
            '项V-s是指集合V-{s}')
        print('隐含集合记号常可以简化有关流的等式.下列引理给出了有关流和隐含记号的几个恒等式')
        print('引理26.1 设G=(V,E)是一个流网络,f是G中的一个流.那么下列等式成立')
        print(' 1) 对所有X∈V,f(X,X)=0')
        print(' 2) 对所有X,Y∈V,f(X,Y)=-f(Y,X)')
        print(' 3) 对所有X,Y,Z∈V,其中X∧Y!=None,有f(X∨Y,Z)=f(X,Y)+f(Y,Z)且f(Z,X∨Y)=f(Z,X)+f(Z,Y)')
        print('作为应用隐含求和记法的一个例子,可以证明一个流的值为进入汇点的全部网络流,即|f|=f(V,t)')
        print('根据流守恒特性,除了源点和汇点以外,对所有顶点来说,进入顶点的总的正流量等于离开该顶点的总的正能量',
            '根据定义,源点顶点总的净流量大于0；亦即，对源点顶点来说，离开它的正流要比进入它的正流更多',
            '对称地，汇点顶点是唯一一个其总的净流量小于0的顶点;亦即,进入它的正流要比离开它的正流更多')
        print('练习26.1-1 利用流的定义，证明如果(u,v)∉E,且(v,u)∉E,有f(u,v)=f(v,u)=0')
        print('练习26.1-2 证明：对于任意非源点非汇点的顶点v,进入v的总正向流必定等于离开v的总正向流')
        print('练习26.1-3 证明在具有多个源点和多个汇点的流网络中,任意流均对应于通过增加一个超级源点',
            '和超级汇点所得到的具有相同值的一个单源点单汇点流')
        print('练习26.1-4 证明引理26.1')
        print('练习26.1-5 在所示的流网络G=(V,E)和流f,找出两个子集合X,Y∈V,且满足f(X,Y)=-f(V-X,Y)',
            '再找出两个子集合X,Y∈V,且满足f(X,Y)!=-f(V-X,Y)')
        print('练习26.1-6 给定流网络G=(V,E),设f1和f2为V*V到R上的函数.流的和f1+f2是从V*V到R上的函数',
            '定义如下：对所有u,v∈V (f1+f2)(u,v)=f1(u,v)+f2(u,v)',
            '如果f1和f2为G的流,则f1+f2必满足流的三条性质中的哪一条')
        print('练习26.1-7 设f为网络中的一个流,a为实数。标量流之积是一个从V*V到R上的函数,定义为(af)(u,v)=a*f(u,v)',
            '证明网络中的流形成一个凸集','即证明如果f1和f2是流,则对所有0<=a<=1,af1+(1-a)f2也是流')
        print('练习26.1-8 将最大流问题表述为一个线性规划问题')
        print('练习26.1-9 略')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_2:
    '''
    chpater26.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.2 note

        Example
        ====
        ```python
        Chapter26_2().note()
        ```
        '''
        print('chapter26.2 note as follow')  
        print('解决最大流问题的Ford-Fulkerson方法,包含具有不同运行时间的几种实现')
        print('Ford-Fulkerson方法依赖于三种重要思想')
        print(' 残留网络,增广路径,割')
        print('这些思想是最大流最小割定理的精髓,该定理用流网络的割来描述最大流的值')
        print('Ford-Fulkerson方法是一种迭代方法.开始时,对所有u,v∈V,有f(u,v)=0,即初始状态时流的值为0',
            '在每次迭代中,可通过寻找一条\"增广路径\来增加流的值"。增广路径可以看作是从源点s到汇点t之间的一条路径',
            '沿该路径可以压入更多的流,从而增加流的值.反复进行这一过程,直至增广路径都被找出为止',
            '最大流最小割定理将说明在算法终止时,这一过程可产生出最大流')
        print('残留网络')
        print('  直观上,给定流网络和一个流,其残留网络由可以容纳更多网络流的边所组成',
            '更形式地,假定有一个网络G=(V,E),其源点为s,汇点到t.设f为G中的一个流,并考察一对顶点u,v∈V',
            '在不超过容量c(u,v)的条件下,从u到v之间可以压入的额外网络流量,就是(u,v)的残留容量(residual capacity),由下式定义:',
            'cf(u,v)=c(u,v)-f(u,v)')
        print('  例如,如果c(u,v)=16且f(u,v)=11,则在不超过边(u,v)的容量限制的条件下,可以再传输cf(u,v)=5个单位的流来增加f(u,v)',
            '当网络流f(u,v)为负值时,残留容量cf(u,v)大于容量c(u,v)')
        print('  例如,如果c(u,v)=16且f(u,v)-4,残留容量cf(u,v)为20')
        print('  解释：从v到u存在着4个单位的网络流,可以通过从u到v压入4个单位的网络来抵消它',
            '然后,在不超过边(u,v)的容量限制的条件下,还可以从u到v压入另外16个单位的网络流',
            '因此,从开始的网络流f(u,v)-4,共压入了额外的20个单位的网络流,并不会超过容量限制')
        print('  在残留网络中,每条边(或称为残留边)能够容纳一个严格为正的网络流')
        print('Ef中的边既可以是E中的边,也可以是它们的反向边',
            '如果边(u,v)∈E有f(u,v)<c(u,v),那么cf(u,v)=c(u,v)-f(u,v)>0且(u,v)属于Ef')
        print('只有当两条边(u,v)和(v,u)中,至少有一条边出现于初始网络中时,边(u,v)才能够出现在残留网络中,所以有如下限制条件:',
            '|Ef|<=2|E|.残留网络Gf本身也是一个流网络,其容量由cf给出.下列引理说明残留网络中的流与初始网络中的流有何关系')
        print('引理26.2 设G=(V,E)是源点为s,汇点为t的一个流网络,且f为G中的一个流',
            '设Gf是由f导出的G的残留网络,且f’为Gf中的一个流,其值|f+f`|=|f|+|f`|')
        print('增广路径')
        print('  已知一个流网络G=(V+E)和流f,增广路径p为残留网络Gf中从s到t的一条简单路径',
            '根据残留网络的定义,在不违反边的容量限制条件下,增广路径上的每条边(u,v)可以容纳从u到v的某额外正网络流')
        print('引理26.3 设G=(V,E)是一个网络流,f是G的一个流,并设p是Gf中的一条增广路径.',
            '用下式定义一个函数：fp：V*V->R')
        print('fp(u,v)=cf(p);fp(u,v)=-cf(p);fp(u,v)=0')
        print('则fp是Gf上的一个流,其值为|fp|=cf(p)>0')
        print('推论26.4 设G=(V,E)是一个流网络,f是G的一个流,p是Gf中的一条增广路径')
        print('流网络的割')
        print('  Ford-Fulkerson方法沿增广路径反复增加流,直至找出最大流为止.',
            '要证明的最大流最小割定理：一个流是最大流,当且仅当它的残留网络不包含增广路径')
        print('流网络G=(V,E)的割(S,T)将V划分成S和T=V-S两部分,使得s∈S,t∈T')
        print('一个网络的最小割也就是网络中所有割中具有最小容量的割')
        print('引理26.5 设f是源点s,汇点为t的流网络G中的一个流.并且(S,T)是G的一个割',
            '则通过割(S,T)的净流f(S,T)=|f|')
        print('推论26.6 对一个流网络G中任意流f来说,其值的上界为G的任意割的容量')
        print('定理26.7(最大流最小割定理) 如果f是具有源点s和汇点t的流网络G=(V,E)中的一个流,则下列条件是等价的:',
            '1) f是G的一个最大流')
        print('2) 残留网络Gf不包含增广路径')
        print('3) 对G的某个割(S,T),有|f|=c(S,T)')
        print('基本的Ford-Fulkerson算法')
        print('  在Ford-Fulkerson方法的每次迭代中,找出任意增广路径p,并把沿p每条边的流f加上其残留容量cf(p)',
            '在Ford-Fulkerson方法的以下实现中,',
            '通过更新有边相连的每对顶点u,v之间网络流f[u,v],来计算出图G=(V,E)中的最大流')
        print('如果u和v之间在任意方向没有边相连,则隐含地假设f[u,v]=0',
            '假定已经在图中给出,且如果(u,v)∉E,有c(u,v)=0.残留容量cf(u,v)',
            '代码中的符号cf(p)实际上只是存储路径p的残留容量的一个临时变量')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_3:
    '''
    chpater26.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.3 note

        Example
        ====
        ```python
        Chapter26_3().note()
        ```
        '''
        print('chapter26.3 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_4:
    '''
    chpater26.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.4 note

        Example
        ====
        ```python
        Chapter26_4().note()
        ```
        '''
        print('chapter26.4 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_5:
    '''
    chpater26.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.5 note

        Example
        ====
        ```python
        Chapter26_5().note()
        ```
        '''
        print('chapter26.5 note as follow')  
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

chapter26_1 = Chapter26_1()
chapter26_2 = Chapter26_2()
chapter26_3 = Chapter26_3()
chapter26_4 = Chapter26_4()
chapter26_5 = Chapter26_5()

def printchapter26note():
    '''
    print chapter26 note.
    '''
    print('Run main : single chapter twenty-six!')  
    chapter26_1.note()
    chapter26_2.note()
    chapter26_3.note()
    chapter26_4.note()
    chapter26_5.note()

# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
if __name__ == '__main__':  
    printchapter26note()
else:
    pass
