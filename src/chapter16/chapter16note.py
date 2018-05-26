# coding:utf-8
# usr/bin/python3
# python src/chapter16/chapter16note.py
# python3 src/chapter16/chapter16note.py
'''
Class Chapter16_1

Class Chapter16_2

Class Chapter16_3

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
from numpy import *

class Chapter16_1:
    '''
    chpater16.1 note and function
    '''
    def recursive_activity_selector(self, s, f, i, j):
        '''
        递归解决活动选择问题
        '''
        m = i + 1
        while m < j and s[m] < f[i]:
            m = m + 1
        if m < j:
            return [m] + self.recursive_activity_selector(s, f, m, j) 
        else:
            return []

    def greedy_activity_selector(self, s, f):
        '''
        迭代贪心算法解决活动选择问题
        '''
        n = len(s)
        A = [1]
        i = 1
        for m in range(2, n):
            if s[m] >= f[i]:
                A = A + [m]
                i = m
        return A

    def normal_activity_selector(self, s, f):
        '''
        常规for循环解决选择问题
        '''
        A = []
        n = len(s)
        c = zeros((n, n))
        length = zeros(n)
        for k in range(n):
            start = s[k]
            end = f[k]
            c[k][k] = k + 1
            length[k] += (end - start)
            for i in range(k):
                if f[i] < start:
                    start = s[i]
                    c[k][i] = i + 1
                    length[k] += (f[i] - s[i])
            for j in range(k + 1, n):
                if s[j] >= end:
                    end = f[j]
                    c[k][j] = j + 1
                    length[k] += (f[j] - s[j])
        return c, length

    def dp_activity_selector(self, s, f):
        '''
        动态规划解决选择问题
        '''
        n = len(s)
        c = zeros((n, n))
        index = zeros((n, n))
        for step in range(2, n):
            for i in range(0, n - 1):
                j = step + i
                if j < n:
                    if f[i] <= s[j]:
                        for k in range(i + 1, j):
                            if f[k] > s[j] or s[k] < f[i]:
                                continue
                            result = c[i][k] + c[k][j] + 1
                            if result > c[i][j]:
                                c[i][j] = result
                                index[i][j] = k
        return index

    def dp_activity_selector_print(self, index, i, j):
        '''
        打印结果
        '''
        k = int(index[i][j])
        if k != 0:
            self.dp_activity_selector_print(index, i, k)
            print(k, end=' ')
            self.dp_activity_selector_print(index, k, j)
        
    def note(self):
        '''
        Summary
        ====
        Print chapter16.1 note

        Example
        ====
        ```python
        Chapter16_1().note()
        ```
        '''
        print('chapter16.1 note as follow')   
        print('第16章 贪心算法')
        # !适用于最优化问题的算法往往具有一系列步骤，每一步有一组选择
        print('适用于最优化问题的算法往往具有一系列步骤，每一步有一组选择')
        print('对许多最优化的问题来说，采用动态规划有点大材小用的意思，只要采用另一些更简单的方法就行了')
        # !贪心算法使得所做的选择在当前看起来都是最佳的，期望通过所做的局部最优得到最终的全局最优
        print('贪心算法使得所做的选择在当前看起来都是最佳的，期望通过所做的局部最优得到最终的全局最优')
        print('贪心算法对大多数最优化问题都能产生最优解，但也不一定总是这样的')
        print('在讨论贪心算法前，首先讨论动态规划方法，然后证明总能用贪心的选择得到其最优解')
        print('16.2给出一种证明贪心算法正确的方法')
        # !有许多被视为贪心算法应用的算法，如最小生成树，Dijkstra的单源最短路径，贪心集合覆盖启发式
        print('有许多被视为贪心算法应用的算法，如最小生成树，Dijkstra的单源最短路径，贪心集合覆盖启发式')
        print('16.1 活动选择问题')
        # !活动选择问题：对几个互相竞争的活动进行调度,它们都要求以独占的方式使用某一公共资源
        print('对几个互相竞争的活动进行调度,它们都要求以独占的方式使用某一公共资源')
        print('设有n个活动和某一个单独资源构成的集合S={a1,a2,...,an}，该资源一次只能被一个活动占用')
        print('各活动已经按照结束时间的递增进行了排序')
        print('每个活动ai都有一个开始时间si和一个结束时间fi，资源一旦被活动ai选中后')
        print('活动ai就开始占据左闭右开时间区间，如果两个活动ai，aj的时间没有交集，称ai和aj是兼容的')
        # !活动选择问题就是要选择出一个由互相兼容的问题组成的最大子集合
        print('动态规划方法解决活动选择问题时，将原方法分为两个子问题，',
            '然后将两个子问题的最优解合并成原问题的最优解')
        # !贪心算法只需考虑一个选择(贪心的选择)，再做贪心选择时，子问题之一必然是空的，因此只留下一个非空子问题
        print('贪心算法只需考虑一个选择(贪心的选择)，再做贪心选择时，子问题之一必然是空的，因此只留下一个非空子问题')
        print('因此找到一种递归贪心算法解决活动选择问题')
        print('活动选择问题的最优子结构')
        print(' 首先为活动选择问题找到一个动态规划解，找到问题的最优子结构。')
        print(' 然后利用这一结构，根据子问题的最优解来构造出原问题的一个最优解')
        print(' 定义一个合适的子问题空间Sij是S中活动的子集，其中，每个活动都在活动ai结束之后开始，且在活动aj开始之前结束')
        print(' 实际上，Sij包含了所有与ai和aj兼容的活动，并且与不迟于ai结束和不早于aj开始的活动兼容')
        print(' 假设活动已按照结束时间的单调递增顺序排序：')
        print(' 一个非空子问题Sij的任意解中必包含了某项活动ak，而Sij的任一最优解中都包含了其子问题实例Sik和Skj的最优解。')
        print(' 因此，可以可以构造出Sij中的最大兼容活动子集。',
            '将问题分成两个子问题(找出Sik和Skj的最大兼容活动子集)，找出这些子问题的最大兼容活动子集Aik和Akj',
            '而后形成最大兼容活动子集Aij如:Aij=Aik ∪ {ak} ∪ Akj')
        print(' 整个问题的一个最优解也是S0,n+1的一个解')
        print('一个递归解')
        print(' 动态规划方法的第二步是递归地定义最优解的值。对于活动选择问题')
        print(' 考虑一个非空子集Sij,如果ak在Sij的最大兼容子集中被使用，则子问题Sik和Skj的最大兼容子集也被使用')
        print(' 递归式：c[i, j] = max{c[i, k] + c[k, j] + 1} 如果Sij不是空集')
        print(' 递归式：c[i, j] = 0 如果Sij是空集')
        print('将动态规划解转化为贪心解')
        print(' 到此为止，写出一个表格化的、自底向上的、基于递归式的动态规划算法是不难的')
        i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]
        s = [0, 1, 3, 0 ,5, 3, 5, 6, 8, 8, 2, 12]
        f = [0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]
        print(' 定理16.1 对于任意非空子问题Sij,设am是Sij中具有最早结束时间的活动:fm=min{fk:ak属于Sij}')
        print(' 那么(1) 活动am在Sij的某最大兼容活动子集中被使用')
        print(' (2)子问题Sim为空，所以选择am将使子问题Smj为唯一可能非空的子问题')
        print('递归贪心算法')
        print(' 介绍一种纯贪心的，自顶向下(递归)的算法解决活动选择问题,',
            '假设n个输入活动已经按照结束时间的单调递增顺序排序。否则可以在O(nlgn)时间内将它们以此排序')
        print(self.recursive_activity_selector(s, f, 0, len(s)))
        print(self.greedy_activity_selector(s, f))
        s = [1, 2, 0 ,5, 3, 5, 6, 8, 8, 2, 12]
        f = [4, 5, 6, 7, 8, 9, 10,11,12,13,14]
        print(self.normal_activity_selector(s, f))
        print('练习16.1-1 活动选择问题的动态规划算法')
        s = [0, 1, 3, 0 ,5, 3, 5, 6, 8, 8, 2, 12, math.inf]
        f = [0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14, math.inf]
        index = self.dp_activity_selector(s, f)
        print(index)
        self.dp_activity_selector_print(index, 0, len(s) - 1)
        print('')
        print('练习16.1-2 略')
        print('练习16.1-3 区间图着色问题：可作出一个区间图，其顶点为已知的活动，其边连接着不兼容的活动',
            '其边连接着不兼容的活动。为使任两个相邻结点的颜色均不相同，',
            '所需的最少颜色数对应于找出调度给定的所有活动所需的最小教室数')
        print('练习16.1-4 并不是任何用来解决活动选择问题的贪心算法都能给出兼容活动的最大集合')
        print(' 请给出一个例子，说明那种在与已选出的活动兼容的活动中选择生存期最短的方法是行不通的')
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

class Chapter16_2:
    '''
    chpater16.2 note and function
    '''
    def zero_one_knapsack_problem_dp(self, total_weight, item_weight, item_value):
        '''
        0-1背包问题的动态规划方法

        时间复杂度`O(n × total_weight)`

        空间复杂度`O(n × total_weight)`

        Args
        ===
        `total_weight` : 背包能容纳的物品总重量

        `item_weight` : `list` 各物品的重量

        `item_value` : `list` 各物品的价值

        Return
        ===
        `item_index` : 存放入背包的物品索引，一个物品只能存放一次

        Example
        ===
        ```python
        >>> total_weight = 50
        >>> item_weight = [10, 20, 30]
        >>> item_value = [60, 100, 120]
        >>> zero_one_knapsack_problem_dp(total_weight, item_weight, item_value):
        >>> [1, 2]
        ```
        '''
        # 动态规划第一步先选取最优子问题结构,并确立表格
        n = len(item_value) 
        W = total_weight
        V = zeros((n, W + 1))
        w = item_weight
        v = item_value   
        for i in range(1, n):
            for j in range(1, W + 1):
                if j < w[i]:
                    V[i][j] = V[i - 1][j]
                else:
                    V[i][j] = max(V[i - 1][j], V[i - 1][j - w[i]] + v[i])
        item = []
        self.__find_zero_one_knapsack_problem_dp_result(V, w, v, n - 1, W, item)
        return item
    
    def __find_zero_one_knapsack_problem_dp_result(self, V, w, v, i, j, item : list):
        if i >= 0:
            if V[i][j] == V[i - 1][j]:
                self.__find_zero_one_knapsack_problem_dp_result(V, w, v, i - 1, j, item)
            elif j - w[i] >= 0 and V[i][j] == V[i - 1][j - w[i]] + v[i]:
                item.append(i)
                self.__find_zero_one_knapsack_problem_dp_result(V, w, v, i - 1, j - w[i], item)

    def partof_knapsack_problem_ga(self, total_weight, item_weight, item_value):
        '''
        部分背包问题的贪心算法

        Args
        ===
        `total_weight` : 背包能容纳的物品总重量

        `item_weight` : `list` 各物品的重量

        `item_value` : `list` 各物品的价值

        Return
        ===
        `item_index` : 存放入背包的物品索引，一个物品只能存放一次

        Example
        ===
        ```python
        >>> total_weight = 50
        >>> item_weight = [10, 20, 30]
        >>> item_value = [60, 100, 120]
        >>> partof_knapsack_problem_ga(total_weight, item_weight, item_value):
        ```
        '''
        w = item_weight
        v = item_value
        n = len(w)
        r = []
        m = total_weight
        for i in range(n):
            r.append(v[i] * 1.0 / w[i])
        # 冒泡排序
        for i in range(1, n):
            for j in range(n - i):
                # 排序
                if r[j] < r[j + 1]:
                    r[j], r[j + 1] = r[j + 1], r[j]
                    w[j], w[j + 1] = w[j + 1], w[j]
                    v[j], v[j + 1] = v[j + 1], v[j]    
        i = 0 
        while m > 0:
            if w[i] <= m:
                m -= w[i]
                print('value:{} weight:{}'.format(v[i], w[i]))
                i += 1
            else:
                print('value:{} weight:{}'.format(v[i], m))
                m = 0

    def cal_compose_value(self, A, B):
        '''
        计算组合价值
        '''
        assert len(A) == len(B)
        n = len(A)
        value = 0
        for i in range(n):
            value += A[i] ** B[i]
        return value

    def insertsort(self, array, start ,end, isAscending=True):
        '''
        Summary
        ===
        插入排序的升序排列(带排序索引), 原地排序
        
        Parameter
        ===
        `array` : a list like

        `start` : sort start index

        `end` : sort end index

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> array = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().insert(array, 1, 4)
        >>> [6 ,2, 3, 4, 5, 1]
        ```
        '''
        if isAscending == True:
            A = array
            for j in range(start + 1, end + 1):
                ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
                # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
                # 后来摸的牌
                key = A[j]
                # 之前手中的已经排序好的牌的最大索引
                i = j - 1
                # 开始寻找插入的位置并且移动牌
                while(i >= 0 and A[i] > key):
                    # 向右移动牌
                    A[i + 1] = A[i]
                    # 遍历之前的牌
                    i = i - 1
                # 后来摸的牌插入相应的位置
                A[i + 1] = key
                # 输出升序排序后的牌
        else:
            A = array
            for j in range(start + 1, end + 1):
                ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
                # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
                # 后来摸的牌
                key = A[j]
                # 之前手中的已经排序好的牌的最大索引
                i = j - 1
                # 开始寻找插入的位置并且移动牌
                while(i >= 0 and A[i] <= key):
                    # 向右移动牌
                    A[i + 1] = A[i]
                    # 遍历之前的牌
                    i = i - 1
                # 后来摸的牌插入相应的位置
                A[i + 1] = key
                # 输出升序排序后的牌
        return A

    def max_compose_value(self, A, B):
        ''' 
        最大化报酬问题，对集合`A` 和 集合`B`排序后，使价值最大 (贪心求解)

        value = argmax(∏ ai ** bi)

        '''
        assert len(A) == len(B)
        n = len(A)
        for i in range(n):
            A = self.insertsort(A, i, n - 1, isAscending=False)
            if A[i] >= 1:
                B = self.insertsort(B, i, n - 1, isAscending=False)  
            else:
                B = self.insertsort(B, i, n - 1, isAscending=True)  
        return self.cal_compose_value(A, B)

    def note(self):
        '''
        Summary
        ====
        Print chapter16.2 note

        Example
        ====
        ```python
        Chapter16_2().note()
        ```
        '''
        print('chapter16.2 note as follow')
        print('16.2 贪心策略的基本内容')
        print('贪心算法是通过做一系列的选择来给出某一问题的最优解。',
            '对算法中的每一个决策点，做一个当时(看起来)是最佳的选择')
        print('这种启发式策略并不是总能产生出最优解，但是常常能给出最优解')
        print('开发一个贪心算法遵循的过程：')
        print(' 1) 决定问题的最优子结构')
        print(' 2) 设计出一个递归解')
        print(' 3) 在递归的任一阶段，最优选择之一总是贪心选择。那么做贪心选择总是安全的')
        print(' 4) 证明通过做贪心选择，所有子问题(除一个以外)都为空')
        print(' 5) 设计出一个实现贪心策略的递归算法')
        print(' 6) 将递归算法转换成迭代算法')
        print('通过这些步骤，可以清楚地发现动态规划是贪心算法的基础。')
        print('实际在设计贪心算法时，经常简化以上步骤，通常直接作出贪心选择来构造子结构，',
            '以产生一个待优化解决的子问题')
        print('无论如何，在每一个贪心算法的下面，总会有一个更加复杂的动态规划解。')
        print('可根据如下的步骤来设计贪心算法：')
        print(' 1) 将优化问题转化成这样的一个问题，即先做出选择，再解决剩下的一个子问题')
        print(' 2) 证明原问题总是有一个最优解是做贪心选择得到的，从而说明贪心选择的安全')
        print(' 3) 说明在做出贪心选择后，剩余的子问题具有这样一个性质。',
            '即如果将子问题的最优解和所做的贪心选择联合起来，可以得出原问题的一个最优解')
        # !贪心算法一般不能够解决一个特定的最优化问题，但是贪心选择的性质和最优子结构时两个关键的特点
        print('贪心算法一般不能够解决一个特定的最优化问题，但是贪心选择的性质和最优子结构时两个关键的特点')
        print('如果能够证明问题具有贪心选择性质和最优子结构，那么就可以设计出它的一个贪心算法')
        print('贪心选择性质：')
        print(' 一个全局最优解可以通过局部最优(贪心)选择来达到。换句话说，当考虑做何选择时，',
            '只考虑对当前问题最佳的选择而不考虑子问题的结果')
        print(' 贪心算法不同于动态规划之处。在动态规划中，每一步都要做出选择，但是这些选择依赖于子问题的解')
        print(' 解动态规划问题一般是自底向上，从小子问题处理至大子问题')
        print(' 在贪心算法中，所做的总是当前看似最佳的选择，然后再解决选择之后所出现的子问题')
        print(' 贪心算法所做的当前选择可能要依赖于已经做出的所有选择，',
            '但不依赖于有待于做出的选择或子问题的解')
        print(' 因此，不像动态规划方法那样自底向上地解决子问题，',
            '贪心策略通常是自顶向下地做的，一个一个地做出贪心选择，不断地将给定的问题实例归约为更小的问题')
        print(' 必须证明在每一步所做的贪心选择最终能产生一个全局最优解，这也是需要技巧的所在')
        print(' 贪心选择性质在面对子问题做出选择时，通常能帮助我们提高效率。例如，在活动选择问题中，',
            '假设已将活动按结束时间的单调递增顺序排序，则每个活动只需检查一次。')
        print(' 通常对数据进行处理或选用合适的数据结构(优先队列)，能够使贪心选择更加快速，因而产生出一个高效的算法')
        print('最优子结构')
        print(' 对一个问题来说，如果它的一个最优解包含了其子问题的最优解，则称该问题具有最优子结构')
        print(' 贪心算法中使用最优子结构时，通常是用更直接的方式。结社在原问题中作了一个贪心选择而得到了一个子问题')
        print(' 真正要做的是证明将此子问题的最优解与所做的贪心选择合并后，的确可以得到原问题的一个最优解')
        print(' 这个方案意味着要对子问题采用归纳法，来证明每个步骤中所做的贪心选择最终会产生出一个最优解')
        print('贪心法与动态规划都利用了最优子结构性质')
        print('0-1背包问题是这样的，有一个贼在偷窃一家商店时发现有n件物品；第i件物品值vi元，重wi磅，此处v和w都是整数')
        print('希望带走的东西越值钱越好，但他的背包至多只能装下W磅的东西,要使价值最高，应该带走哪几样东西')
        print('部分背包问题是在0-1背包问题的基础上可以选择带走物品的一部分')
        # !虽然部分0-1背包问题和部分背包问题特别相似，但是部分背包问题可以用贪心策略来解决，而0-1背包问题却不行
        print('虽然部分0-1背包问题和部分背包问题特别相似，但是部分背包问题可以用贪心策略来解决，而0-1背包问题却不行')
        print('使用贪心算法解决部分背包问题：先对每件物品计算其每磅的价值vi/wi。按照一种贪心策略，',
            '窃贼开始时对具有最大每磅价值的物品尽量多拿一些。如果他拿完了该物品而仍然可以取一些其他物品时，',
            '他就再取具有次大的每磅价值的物品，一直继续下去，直到不能再取为止。这样，通过按每磅价值来对所有物品排序',
            '贪心算法就可以O(nlgn)时间运行。关于部分背包问题具有贪心选择性质的证明')
        print('一个简单的问题可以说明贪心为什么不适用0-1背包问题，背包能承受的最大重量为50磅')
        print('物品1 重10磅 值60元(每磅6元)；物品2 重20磅 值100元(每磅5元)；物品3 重30磅 值120元(每磅4元)')
        print('按照贪心策略(即只关注当前的最优情况)，就要取物品1，然而最优解是一定不能取物品1的')
        print('最优解取的是物品2和物品3，留下物品1.两种包含物品1的可能解都是次优的')
        print('即贪心策略对0-1背包问题不适用')
        print('然而对于部分背包问题，在按照贪心策略先取物品1以后，确实可以产生一个最优解')
        print('在0-1背包问题中不应该取物品1的原因在与这样无法把背包填满，空余的空间就降低了他的货物的有效每磅价值')
        print('在0-1背包问题中，当我们考虑是否要把一件物品加到背包中时，必须对把该问题加进去的子问题的解与不取该物品的子问题的解进行比较')
        print('由这种方式形成的问题导致了许多重叠子问题(这是动态规划的一个特点)，所以，可以用动态规划来解决0-1背包问题')
        print('练习16.2-1 证明部分背包问题具有贪心选择性质')
        print('练习16.2-2 请给出一个解决0-1背包问题的运行时间为O(n W)的动态规划方法，',
            'n为物品件数，W为窃贼可放入他背包物品的最大重量')
        # 一般动态规划在输入数据中填入首项0
        total_weight = 8
        item_weight = [0, 5, 4, 3, 1]
        item_value = [0, 3, 4, 5, 6]
        print(self.zero_one_knapsack_problem_dp(total_weight, item_weight, item_value))
        total_weight = 8
        item_weight = [2, 3, 4, 5]
        item_value = [3, 4, 5, 6]
        print(self.zero_one_knapsack_problem_dp(total_weight, item_weight, item_value))
        print('贪心算法解部分背包问题')
        self.partof_knapsack_problem_ga(total_weight, item_weight, item_value)
        print('练习16.2-3 从价值高重量轻的开始拿，拿到满为止')
        print('练习16.2-4 略,公路加油问题')
        print('练习16.2-5 请描述一个算法，使之对给定的一实数轴上的点集{x1,x2,...,xn},能确定包含所有',
            '给定点的最小单位闭区间闭集合')
        print('练习16.2-6 带权中位数：在O(nlgn)的最坏情况时间内求出n个元素的带权中位数',
            '说明如何在O(n)时间内解决部分背包问题')
        print('练习16.2-7 最大化报酬问题')
        A = [1.1, 0.2, 3, 4, 5]
        B = [4.3, 4, 3, 2, 1]
        print(self.cal_compose_value(A, B))
        print(self.max_compose_value(A, B))
        print(A)
        print(B)
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

chapter16_1 = Chapter16_1()
chapter16_2 = Chapter16_2()

def printchapter16note():
    '''
    print chapter16 note.
    '''
    print('Run main : single chapter sixteen!')  
    chapter16_1.note()
    chapter16_2.note()

# python src/chapter16/chapter16note.py
# python3 src/chapter16/chapter16note.py
if __name__ == '__main__':  
    printchapter16note()
else:
    pass
