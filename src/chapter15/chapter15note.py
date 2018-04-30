
# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
'''
Class Chapter15_1

Class Chapter15_2

Class Chapter15_3

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
from numpy import array as _array
from numpy import *

class Chapter15_1:
    '''
    chpater15.1 note and function
    '''
    index1 = 0
    index2 = 1
    f = [[], []]
    l = [[], []]
    def fastway(self, a, t, e, x, n):
        '''
        计算最快时间 Θ(n)

        Args
        ===
        `a` : `a[i][j]` 表示在第`i`条装配线`j`装配站的装配时间

        `t` : `t[i][j]` 表示在第`i`条装配线`j`装配站移动到另外一条装配线所需要的时间

        `e` : `e[i]` 表示汽车底盘进入工厂装配线`i`所需要的时间

        `x` : `x[i]` 表示完成的汽车花费离开装配线所需要的时间

        `n` : 每条装配线所具有的装配站数量

        Return
        ===
        `(fxin, lxin)` : a tuple like

        Example
        ===
        ```python
        a = [[7, 9, 3, 4, 8, 4], [8, 5, 6, 4, 5, 7]]
        t = [[2, 3, 1, 3, 4], [2, 1, 2, 2, 1]]
        e = [2, 4]
        x = [3, 2]
        n = 6
        self.fastway(a, t, e, x, n)
        >>> (38, 0)
        ```

        '''
        # 定义最优解变量
        ## 路径最优解
        lxin = 0
        ## 时间最优解
        fxin = 0
        # 定义两条装配线
        index1 = self.index1
        index2 = self.index2
        # 子问题存储空间
        f = self.f
        l = self.l
        # 开辟空间存储动态规划子问题的解
        f[index1] = list(range(n))
        f[index2] = list(range(n))
        l[index1] = list(range(n))
        l[index2] = list(range(n))
        # 上装配线
        f[index1][0] = e[index1] + a[index1][0]
        f[index2][0] = e[index2] + a[index2][0]
        # 求解子问题
        for j in range(1, n):
            # 求解装配线1的子问题,因为求解最短时间，谁小赋值谁
            if f[index1][j - 1] + a[index1][j] <= f[index2][j - 1] + t[index2][j - 1] + a[index1][j]:
                f[index1][j] = f[index1][j - 1] + a[index1][j]
                l[index1][j] = index1
            else:
                f[index1][j] = f[index2][j - 1] + t[index2][j - 1] + a[index1][j]
                l[index1][j] = index2
            # 求解装配线1的子问题,因为求解最短时间，谁小赋值谁
            if f[index2][j - 1] + a[index2][j] <= f[index1][j - 1] + t[index1][j - 1] + a[index2][j]:
                f[index2][j] = f[index2][j - 1] + a[index2][j]
                l[index2][j] = index2
            else:
                f[index2][j] = f[index1][j - 1] + t[index1][j - 1] + a[index2][j]
                l[index2][j] = index1
        n = n - 1
        # 求解离开装配线时的解即为总问题的求解，因为子问题已经全部求解
        if f[index1][n] + x[index1] <= f[index2][n] + x[index2]:
            fxin = f[index1][n] + x[index1]
            lxin = index1
        else:
            fxin = f[index2][n] + x[index2]
            lxin = index2
        # 返回最优解
        return (fxin, lxin)

    def printstations(self, l, lxin, n):
        '''
        打印最优通过的路线
        '''
        index1 = self.index1
        index2 = self.index2
        i = lxin - 1
        print('line', i + 1, 'station', n)
        for j in range(2, n + 1):
            m = n - j + 2 - 1
            i = l[i][m]
            print('line', i + 1, 'station', m)

    def __printstations_ascending(self, l, i, m):
        if m - 1 <= 0:
            print('line', i + 1, 'station', m)
        else:
            self.__printstations_ascending(l, l[i][m - 1], m - 1)
        print('line', i + 1, 'station', m)
        
    def printstations_ascending(self, l, lxin, n):
        '''
        升序打印最优通过的路线(递归方式)
        '''
        index1 = self.index1
        index2 = self.index2
        _lxin = lxin - 1
        self.__printstations_ascending(l, _lxin, n)

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
        # !动态规划适用于问题可以分解为若干子问题,关键技术是存储这些子问题每一个解，以备它重复出现
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
        print('步骤1.通过工厂最快路线的结构,子问题最优结果结果的存储空间')
        print(' 动态规划方法的第一个步骤是描述最优解的结构的特征。对于装配线调度问题，可以如下执行。')
        print(' 首先，假设通过装配站S1,j的最快路线通过了装配站S1,j-1。关键的一点是这个底盘必定是利用了最快的路线从开始点到装配站S1,j-1的')
        print(' 更一般地，对于装配线调度问题，一个问题的最优解包含了子问题的一个最优解。')
        print(' 我们称这个性质为最优子结构，这是是否可以应用动态规划方法的标志之一')
        print(' 为了寻找通过任一条装配线上的装配站j的最快路线，我们解决它的子问题，即寻找通过两条装配线上的装配站j-1的最快路线')
        print(' 所以，对于装配线调度问题，通过建立子问题的最优解，就可以建立原问题某个实例的一个最优解')
        print('步骤2.一个递归的解，总时间最快就是子问题最快')
        print(' 在动态规划方法中，第二个步骤是利用子问题的最优解来递归定义一个最优解的值。')
        print(' 对于装配线的调度问题，选择在两条装配线上通过装配站j的最快路线的问题来作为子问题')
        print(' j=1,2,...,n。令fi[j]表示一个底盘从起点到装配站Sij的最快可能时间')
        print(' 最终目标是确定底盘通过工厂的所有路线的最快时间，记为f')
        print(' 底盘必须一路径由装配线1或2通过装配站n，然后到达工厂的出口，由于这些路线的较快者就是通过整个工厂的最快路线')
        print(' f=min(f1[n]+x1,f2[n]+x2)')
        print(' 要对f1[1]和f2[1]进行推理也是容易的。不管在哪一条装配线上通过装配站1，底盘都是直接到达该装配站的')
        print('步骤3.计算最快时间fxin')
        print(' 此时写出一个递归算法来计算通过工厂的最快路线是一件简单的事情，',
            '这种递归算法有一个问题：它的执行时间是关于n的指数形式')
        # !装配站所需时间
        a = [[7, 9, 3, 4, 8, 4],\
             [8, 5, 6, 4, 5, 7]]
        # !装配站切换到另一条线花费的时间
        t = [[2, 3, 1, 3, 4],\
             [2, 1, 2, 2, 1]]
        # !进入装配线所需时间
        e = [2, 4]
        # !离开装配线所需时间
        x = [3, 2]
        # !每条装配线装配站的数量
        n = 6
        result = self.fastway(a, t, e, x, n)
        fxin = result[0]
        lxin = result[1] + 1
        print('fxin:', fxin, ' lxin:', lxin, 'l[lxin]:', _array(self.l)[lxin - 1] + 1)
        print('最优路径输出(降序从尾到头)')
        self.printstations(self.l, lxin, n)    
        print('存储的子问题的解为：')
        print('f:')
        print(_array(self.f))
        print('l:')
        print(_array(self.l) + 1)
        print('步骤4.构造通过工厂的最快路线lxin')
        print('练习15.1-1 最优路径输出(升序从头到尾)')
        # 通过递归的方式先到达路径头
        self.printstations_ascending(self.l, lxin, n)
        print('练习15.1-2 定理：在递归算法中引用fi[j]的次数ri(j)等于2^(n-j)')
        print('练习15.1-3 定理：所有引用fi[j]的总次数等于2^(n+1)-2')
        print('练习15.1-4 包含fi[j]和li[j]值的表格共含4n-2个表项。说明如何把空间需求缩减到共2n+2')
        print('练习15.1-5 略')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_2:
    '''
    chpater15.2 note and function
    '''

    def matrix_multiply(self, A, B):
        '''
        两个矩阵相乘
        '''
        rowA = shape(A)[0]
        colunmA = shape(A)[1]
        rowB = shape(B)[0]
        colunmB = shape(B)[1]
        C = ones([rowA, colunmB])
        if colunmA != rowA:
            raise Exception('incompatible dimensions')
        else:
            for i in range(rowA):
                for j in range(colunmB):
                    C[i][j] = 0
                    for k in range(colunmA):
                        C[i][j] = C[i][j] + A[i][k] * B[k][j]
            return C

    def matrix_chain_order(self, p):
        '''
        算法：填表`m`的方式对应于求解按长度递增的矩阵链上的加全部括号问题

        Return
        ===
        `(m, s)`

        `m` : 存储子问题的辅助表`m`

        `s` : 存储子问题的辅助表`s`

        Example
        ===
        ```python
        matrix_chain_order([30, 35, 15, 5, 10, 20, 25])
        >>> (m, s)
        ```
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        for i in range(n):
            m[i][i] = 0
        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l - 1
                m[i][j] = math.inf
                for k in range(i, j):
                    q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]          
                    if q < m[i][j]:
                        m[i][j] = q
                        s[i][j] = k + 1
        return (m, s)

    def __print_optimal_parens(self, s, i, j):
        '''
        输出矩阵链乘积的一个最优加全部括号形式
        '''
        i = int(i)
        j = int(j)
        if i == j:
            print('A{}'.format(i + 1), end='')
        else:
            print('(', end='')
            self.__print_optimal_parens(s, i, s[i][j])
            self.__print_optimal_parens(s, s[i][j] + 1, j)
            print(')', end='')

    def print_optimal_parens(self, s):
        '''
        输出矩阵链乘积的一个最优加全部括号形式
        '''
        s = s - 1
        self.__print_optimal_parens(s, 0, shape(s)[-1] - 1)

    def __matrix_chain_multiply(self, A, s, i, j):
        pass

    def matrix_chain_multiply(self, A : list):
        '''
        调用矩阵链乘法对矩阵数组进行连乘
        '''
        p = []
        for a in A:
            row = shape(a)[0]
            p.append(row)
        p.append(shape(A[-1])[1])
        m, s = self.matrix_chain_order(p)
        return self.__matrix_chain_multiply(A, s, 1, len(p) - 1)

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
        print('15.2 矩阵链乘法')
        print('例如：如果矩阵链为A B C D')
        print('则乘积ABCD可用五种不同的方式加括号')
        print('(A(B(CD)))')
        print('(A((BC)D))')
        print('((AB)(CD))')
        print('((A(BC))D)')
        print('(((AB)C)D)')
        A = [[1, 2], [3, 4]]
        print(self.matrix_multiply(A, A))
        A = array(A)
        print(A * A)
        print('为了计算矩阵链乘法，可将两个矩阵相乘的标准算法作为一个子程序Θ(n^3)，矩阵乘法满足结合律')
        print('矩阵链乘法加括号的顺序对求积运算的代价有很大的影响。')
        print('矩阵乘法当且仅当两个矩阵相容(A的列数等于B的行数)，才可以进行相乘运算')
        print('C(p,r) = A(p,q) * B(q,r)')
        print('注意在矩阵链乘法当中，实际上并没有把矩阵相乘，目的仅是确定一个具有最下代价的相乘顺序')
        print('确定最优顺序花费的时间能在矩阵乘法上得到更好的回报')
        print('计算全部括号的重数')
        print('设P(n)表示一串n个矩阵可能的加全部括号的方案数')
        print('用动态规划求解矩阵链乘法顺序，使用穷举的方式不是很好的一个方式，随着矩阵数量n的增长，')
        print('P(n)的一个类似递归解的Catalan数序列，其增长的形式是Ω(4^n/n^(3/2))')
        print('P(n)递归式的一个解为Ω(2^n),所以解的个是指数形式，穷尽策略不是一个好的形式')
        print('步骤1.最优加全部括号的结构')
        print(' 动态规划方法的第一步是寻找最优的子结构')
        print(' 然后，利用这一子结构，就可以根据子问题的最优解构造出原问题的一个最优解。')
        print(' 对于矩阵链乘法问题，可以执行如下这个步骤')
        print(' 用记号Ai..j表示对乘积AiAi+1Aj求值的结果，其中i<=j,如果这个问题是非平凡的，即i<j')
        print(' 则对乘积AiAi+1...Aj的任何全部加括号形式都将乘积在Ak与Ak+1之间分开，此处k是范围1<=k<j之内的一个整数')
        print(' 就是说，对某个k值，首先计算矩阵Ai..k和Ak+1..j,然后把它们相乘就得到最终乘积Ai..j')
        print(' 这样，加全部括号的代价就是计算Ai..k和Ak+1..j的代价之和再加上两者相乘的代价')
        print('步骤2.一个递归解')
        print(' 接下来，根据子问题的最优解来递归定义一个最优解的代价。对于矩阵链乘法问题，子问题即确定AiAi+1...Aj的加全部括号的最小代价问题')
        print(' 此处1<=i<=j<=n。设m[i,j]为计算矩阵Ai..j所需的标量乘法运算次数的最小值；对整个问题，计算A1..n的最小代价就是m[1,n]')
        print(' 递归定义m[i,j]。如果i==j,则问题是平凡的；矩阵链只包含一个矩阵Ai..i=Ai,故无需做任何标量乘法来计算chengji')
        print(' 关于对乘积AiAi+1...Aj的加全部括号的最小代价的递归定义为')
        print(' m[i,j]=0, i = j;  m[i,j]=min{min[i,k]+m[k+1,j] + pi-1pkpj}, i < j')
        print('步骤3.计算最优代价')
        print(' 可以很容易地根据递归式，来写一个计算乘积A1A2...An的最小代价m[1,n]的递归算法。',
            '然而这个算法具有指数时间，它与检查每一种加全部括号乘积的强力法差不多')
        print(' 但是原问题只有相当少的子问题：对每一对满足1<=i<=j<=n的i和j对应一个问题，总共Θ(n^2)种')
        print(' 一个递归算法在其递归树的不同分支中可能会多次遇到同一个子问题，子问题重叠这一性质')
        print(' 不是递归地解递归式，而是执行动态规划方法的第三个步骤，使用自底向上的表格法来计算最优代价')
        print(' 假设矩阵Ai的维数是pi-1×pi,i=1,2,...,n。输入是一个序列p=<p0,p1,...pn>,其中length[p]=n+1')
        print(' 程序使用一个辅助表m[1..n,1..n]来保存m[i,j]的代价')
        print('例子：矩阵链 A1(30 * 35) A2(35 * 15) A3(15 * 5) A4(5 * 10) A5(10 * 20) A6(20 * 25)')
        print('的一个最优加全部括号的形式为((A1(A2A3))((A4A5)A6))')
        p = [30, 35, 15, 5, 10, 20, 25]
        n = len(p) - 1
        m, s = self.matrix_chain_order(p)
        print('the m is ')
        print(m)
        print('the s is ')
        print(s)
        print('最优加全部括号形式为：')
        self.print_optimal_parens(s)
        print('')
        # self.print_optimal_parens(s, 0, n - 1)
        print('步骤4.构造一个最优解')
        print(' 虽然MATRIX—CHAIN-ORDER确定了计算矩阵链乘积所需的标量乘积法次数，但没有说明如何对这些矩阵相乘(如何加全部括号)')
        print(' 利用保存在表格s[1..n,1..n]内的、经过计算的信息来构造一个最优解并不难。')
        print(' 在每一个表项s[i,j]中，记录了对乘积AiAi+1...Aj在Ak与Ak+1之间，进行分裂以取得最优加全部括号时的k值')
        print('练习15.2-1 对6个矩阵维数为<5, 10, 3, 12, 5, 50, 6>的各矩阵，找出其矩阵链乘积的一个最优加全部括号')
        p = [5, 10, 3, 12, 5, 50, 6]
        n = len(p) - 1
        m, s = self.matrix_chain_order(p)
        print('the m is ')
        print(m)
        print('the s is ')
        print(s)
        print('最优加全部括号形式为：')
        self.print_optimal_parens(s)
        print('')
        print('练习15.2-2 给出一个矩阵链乘法算法MATRX-CHAIN_MULTIPLY(A, s, i, j), 初始参数为A, s, 1, n')
        print('练习15.2-3 用替换法证明递归公式的解为Ω(2^n)')
        print('练习15.2-4 设R(i, j)表示在调用MATRIX-CHAIN—ORDER中其他表项时，',
            '表项m[i, j]被引用的次数(n^3-n)/3')
        print('练习15.2-5 定理：一个含n个元素的表达式的加全部括号中恰有n-1对括号(显然n个数的乘法做n-1次两数相乘即可出结果)')
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
        print('15.3 动态规划基础')
        print('从工程的角度看，什么时候才需要一个问题的动态规划解')
        print('适合采用动态规划解决最优化问题的两个要素：最优子结构和重叠子问题')
        print('用备忘录充分利用重叠子问题性质')
        print('最优子结构')
        print(' 用动态规划求解优化问题的第一步是描述最优解的结构')
        print(' 如果一个问题的最优解中包含了子问题的最优解，则该问题具有最优子结构')
        print(' 当一个问题包含最优子结构时，提示我们动态规划是可行的，当然贪心算法也是可行的')
        print('在寻找最优子结构时，可以遵循一种共同的模式')
        print(' 1.问题的一个解可以看作是一个选择')
        print(' 2.假设对一个给定的问题，已知的是一个可以导致最优解的选择')
        print(' 3.在已知这个选择后，要确定哪些子问题会随之发生')
        print(' 4.假设每个子问题的解都不可能是最优的选择，则问题也不可能是最优的')
        print('为了描述子问题空间，尽量保持这个空间简单')
        print('非正式地，一个动态规划算法地运行时间依赖于两个因素地乘积，子问题地总个数和每一个问题有多少种选择')
        print('在装配线调度中，总共有Θ(n)个子问题，并且只有两个选择来检查每个子问题，所以执行时间为Θ(n)。')
        print('对于矩阵链乘法，总共有Θ(n^2)个子问题，在每个子问题中又至多有n-1个选择，因此执行时间为O(n^3)')
        print('动态规划以自底向上的方式来利用最优子结构。首先找到子问题的最优解，解决子问题，然后找到问题的一个最优解')
        print('而贪心算法与动态规划有着很多相似之处。特别地，贪心算法适用的问题也具有最优子结构。')
        # !贪心算法与动态规划有一个显著的区别，就是在贪心算法中，是以自顶向下的方式使用最优子结构
        # !贪心算法会先做选择，在当时看起来是最优的选择，然后再求解一个结果子问题，而不是先寻找子问题的最优解，然后再做选择
        print('贪心算法与动态规划有一个显著的区别，就是在贪心算法中，是以自顶向下的方式使用最优子结构')
        print('贪心算法会先做选择，在当时看起来是最优的选择，然后再求解一个结果子问题，而不是先寻找子问题的最优解，然后再做选择')
        print('注意：在不能应用最优子结构的时候，就一定不能假设它能够应用，已知一个有向图G=(V,E)和结点u,v∈V')
        print('无权最短路径：找出一条从u到v的包含最少边数的路径。这样一条路径必须是简单路径，因为从路径中去掉一个回路后，会产生边数更少的路径')
        print('无权最长简单路径：找出一条从u到v的包含最多边数的简单路径，需要加入简单性需求，否则就可以遍历一个回路任意多次')
        print('这样任何从u到v的路径p必定包含一个中间顶点，比如w，')
        print('对无权最长简单路径问题，假设它具有最优子结构。最终结论：说明对于最长简单路径，不仅缺乏最优子结构，而且无法根据子问题的解来构造问题的一个合法解')
        print('而且在寻找最短路径中子问题是独立的，答案是子问题本来就没有共享资源')
        print('装配站问题和矩阵链乘法问题都有独立的子问题')
        print('重叠子问题')
        print('适用于动态规划求解的最优化问题必须具有的第二个要素是子问题的空间要很小，也就是用来解原问题的递归算法可反复地解同样的子问题，而不是总在产生新的问题')
        print('典型地，不同的子问题数是输入规模的一个多项式。当一个递归算法不断地调用同一问题时，我们说该最优问题包含重叠子问题')
        print('相反地，适合用分治法解决的问题往往在递归的每一步都产生全新的问题。',
            '动态规划算法总是充分利用重叠子问题，即通过每个子问题只解一次，把解保存在一个在需要时就可以查看的表中，而每次查表的时间为常数')
        print('动态规划要求其子问题即要独立又要重叠')
        # !动态规划最好存储子问题的结果在表格中，省时省力
        print('做备忘录')
        print('动态规划有一种变形，它既具有通常的动态规划方法的效率，又采用了一种自顶向下的策略。其思想就是备忘原问题的自然但是低效的递归算法')
        print('像在通常的动态规划中一样，维护一个记录了子问题解的表，但有关填表动作的控制结构更像递归算法')
        print('加了备忘录的递归算法为每一个子问题的解在表中记录一个表项。开始时，每个表项最初都包含一个特殊的值，以表示该表项有待填入')
        print('总之，矩阵链乘法问题可以在O(n^3)时间内，用自顶向下的备忘录算法或自底向上的动态规划算法解决')
        print('两种方法都利用了重叠子问题的性质。原问题共有Θ(n^2)个不同的子问题，这两种方法对每个子问题都只计算一次。',
            '如果不使用做备忘录,则自然递归算法就要以指数时间运行，因为它要反复解已经解过的子问题')
        print('在实际应用中，如果所有的子问题都至少要被计算一次，则一个自底向上的动态规划算法通常要比一个自顶向下的做备忘录算法好出一个常数因子，',
            '因为前者无需递归的代价，而且维护表格的开销也小一点')
        print('此外，在有些问题中，还可以用动态规划算法中的表存取模式来进一步减少时间或空间上的需求')
        print('练习15.3-1 RECURSIVE-MATRIX-CHAIN要比枚举对乘积所有可能的加全部括号并逐一计算其乘法的次数')
        print('练习15.3-2 请解释在加速一个好的分治算法如合并排序方面，做备忘录方法为什么没有效果。',
            '因为分治算法子问题并没有重复和最优，只是一个解的过程。合并排序谁与谁合并已经确定')
        print('练习15.3-3 考虑矩阵链乘法问题的一个变形，其目标是加全部括号矩阵序列以最大化而不是最小化标量乘法的次数。这个问题具有最优子结构')
        print('练习15.3-4 描述装配线调度问题如何具有重叠子问题')
        print('练习15.3-5 在动态规划中，我们先求解各个子问题，我们先求解各个子问题，然后再来决定该选择它们中的哪一个来用在原问题的最优解中。')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py


class Chapter15_4:
    '''
    chpater15.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter15.4 note

        Example
        ====
        ```python
        Chapter15_4().note()
        ```
        '''
        print('chapter15.4 note as follow')   
        print('15.4 最长公共子序列')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

chapter15_1 = Chapter15_1()
chapter15_2 = Chapter15_2()
chapter15_3 = Chapter15_3()
chapter15_4 = Chapter15_4()

def printchapter15note():
    '''
    print chapter15 note.
    '''
    print('Run main : single chapter fiveteen!')  
    chapter15_1.note()
    chapter15_2.note()
    chapter15_3.note()
    chapter15_4.note()

# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
if __name__ == '__main__':  
    printchapter15note()
else:
    pass
