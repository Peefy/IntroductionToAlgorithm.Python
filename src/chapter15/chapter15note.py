
# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
'''
Class Chapter15_1

Class Chapter15_2

Class Chapter15_3

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

class Chapter15_1:
    '''
    chpater15.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter15.1 note

        Example
        ====
        ```python
        Chapter15_1().note()
        ```
        '''
        print('chapter15.1 note as follow')   
        print('第四部分 高级设计和分析技术')
        print('这一部分将介绍设计和分析高效算法的三种重要技术：动态规划(第15章)，贪心算法(第16章)和平摊分析(第17章)')
        print('本书前面三部分介绍了一些可以普遍应用的技术，如分治法、随机化和递归求解')
        print('这一部分的新技术要更复杂一些，但它们对有效地解决很多计算问题来说很有用')
        print('动态规划通常应用于最优化问题，即要做出一组选择以达到一个最优解。',
            '在做选择的同时，经常出现同样形式的子问题。当某一特定的子问题可能出自于多于一种选择的集合时，动态规划非常有效')
        print(' 关键技术是存储这些子问题每一个解，以备它重复出现。第15章说明如何利用这种简单思想，将指数时间的算法转化为多项式时间的算法')
        print('像动态规划算法一样，贪心算法通常也是应用于最优化问题。在这种算法中，要做出一组选择以达到一个最优解。')
        print(' 采用贪心算法可以比用动态规划更快地给出一个最优解。但是不同意判断贪心算法是否一定有效。')
        print(' 第16章回顾拟阵理论，它通常可以用来帮助做出这种判断。')
        print('平摊分析是一种用来分析执行一系列类似操作的算法的工具。',
            '平摊分析不仅仅是一种分析工具，也是算法设计的一种思维方式，',
                '因为算法的设计和对其运行时间的分析经常是紧密相连的')
        print('第15章 动态规划')
        print('和分治法一样，动态规划是通过组合子问题的解而解决整个问题的')
        print('分治法算法是指将问题划分成一些独立的子问题，递归地求解各子问题，然后合并子问题的解而得到原问题的解')
        print('于此不同，动态规划适用于子问题不是独立的情况，也就是各子问题包含的公共的子子问题。')
        print('动态规划不需要像分治法那样重复地求解子子问题，对每个子子问题只求解一次，将其结果保存在一张表中')
        print('动态规划通常应用于最优化问题。此类问题可能有很多种可行解。每个解有一个值，希望找出一个具有最优(最大或最小)值的解')
        print('动态规划算法的设计可以分为如下4个步骤：')
        print(' 1.描述最优解的结构')
        print(' 2.递归定义最优解的值')
        print(' 3.按自底向上的方式计算最优解的值')
        print(' 4.由计算出的结果构造一个最优解')
        print('第1~3步构成问题的动态规划解的基础。第4步在只要求计算最优解的值时可以略去')
        print('15.1 装配线调度')
        print('一个动态规划的例子是求解一个制造问题')
        print('某汽车公司在有两条装配线的工厂内生产汽车，一个汽车底盘在进入每一条装配线后，在一些装配站中会在底盘上安装部件')
        print('然后，完成的汽车在装配线的末端离开。每一条装配线上有n个装配站，编号为j=1,2..,n')
        print('将装配线i(i为1或2)的第j个装配站表示为Si,j。装配线1的第j个站(S1,j)和装配线2的第j个站(S2,j)执行相同的功能')
        print('然而，这些装配站是在不同的时间建造的，并且采用了不同的技术；因此在每个站上所需的时间是不同的')
        print('在不同站所需要的时间为aij,一个汽车底盘进入工厂，然后进入装配线i(i为1或2),花费时间ei.')
        print('在通过一条线的第j个装配站后，这个底盘来到任一条线的第(j+1)个装配站')
        print('如果它留在相同的装配线，则没有移动的开销；但是，如果在装配站Sij后，它移动了另一条线上，则花费时间为tij')
        print('在离开一条线的第n个装配站后，完成的汽车花费时间xi离开工厂。待求解的问题是确定应该在装配线1内选择哪些站、在装配线2内选择哪些站')
        print('才能使汽车通过工厂的总时间最小')
        print('显然，当有很多个装配站时，用强力法(brute force)来极小化通过工厂装配线的时间是不可能的。')
        print('如果给定一个序列，在装配线1上使用哪些站，在装配线2上使用哪些站，则可以在Θ(n)时间内,',
            '很容易计算出一个底盘通过工厂装配线要花的时间')
        print('不幸地是，选择装配站的可能方式有2^n种；可以把装配线1内使用的装配站集合看作{1,2,..,n}的一个子集')
        print('因此，要通过穷举所有可能的方式、然后计算每种方式花费的时间来确定最快通过工厂的路线，需要Ω(2^n)时间，这在n很大时是不行的')
        print('步骤1.通过工厂最快路线的结构')
        print(' 动态规划方法的第一个步骤是描述最优解的结构的特征。对于装配线调度问题，可以如下执行。')
        print(' 首先，假设通过装配站S1,j的最快路线通过了装配站S1,j-1。关键的一点是这个底盘必定是利用了最快的路线从开始点到装配站S1,j-1的')
        print(' 更一般地，对于装配线调度问题，一个问题的最优解包含了子问题的一个最优解。')
        print(' 我们称这个性质为最优子结构，这是是否可以应用动态规划方法的标志之一')
        print(' 为了寻找通过任一条装配线上的装配站j的最快路线，我们解决它的子问题，即寻找通过两条装配线上的装配站j-1的最快路线')
        print(' 所以，对于装配线调度问题，通过建立子问题的最优解，就可以建立原问题某个实例的一个最优解')
        print('步骤2.一个递归的解')
        print('步骤3.计算最快时间')
        print('步骤4.构造通过工厂的最快路线')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_2:
    '''
    chpater15.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter15.2 note

        Example
        ====
        ```python
        Chapter15_2().note()
        ```
        '''
        print('chapter15.2 note as follow')   
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_3:
    '''
    chpater15.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter15.3 note

        Example
        ====
        ```python
        Chapter15_3().note()
        ```
        '''
        print('chapter15.3 note as follow')   
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

chapter15_1 = Chapter15_1()
chapter15_2 = Chapter15_2()
chapter15_3 = Chapter15_3()

def printchapter15note():
    '''
    print chapter15 note.
    '''
    print('Run main : single chapter fiveteen!')  
    chapter15_1.note()
    chapter15_2.note()
    chapter15_3.note()

# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
if __name__ == '__main__':  
    printchapter15note()
else:
    pass
