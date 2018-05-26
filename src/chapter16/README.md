---
layout: post
title: "用Python实现贪心算法解决的部分问题"
description: "用Python实现贪心算法解决的部分问题"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/05/26/
---

## 贪心算法 (Greedy Algorithm)

适用于最优化问题的算法往往具有一系列步骤，每一步有一组选择

贪心算法使得所做的选择在当前看起来都是最佳的，期望通过所做的局部最优得到最终的全局最优

有许多被视为贪心算法应用的算法，如最小生成树，Dijkstra的单源最短路径，贪心集合覆盖启发式

活动选择问题：对几个互相竞争的活动进行调度,它们都要求以独占的方式使用某一公共资源

活动选择问题就是要选择出一个由互相兼容的问题组成的最大子集合

贪心算法只需考虑一个选择(贪心的选择)，再做贪心选择时，子问题之一必然是空的，因此只留下一个非空子问题

### GA和DP的不同

贪心是自下而上 动态规划是自上而下，动态规划的最优解的子集仍然是子问题的最优解，所以存在最优解迭代公式和存放最优解的表格；贪心是从第一步每一步先使子问题达到最优

### 活动选择问题

```python

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

```

### 0-1背包问题
```python
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
```

### 部分背包问题

```python

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

```

### 计算组合价值最大问题(贪心求解)

```python

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

```

[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/blob/master/src/chapter16)
