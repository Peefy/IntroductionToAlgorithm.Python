---
layout: post
title: "用Python实现动态规划解决的部分问题"
description: "用Python实现动态规划解决的部分问题"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/05/05/
---

## 动态规划 (DynamicProgramming)

动态规划通常应用于最优化问题，即要做出一组选择以达到一个最优解

动态规划适用于问题可以分解为若干子问题,关键技术是存储这些子问题每一个解，以备它重复出现

### 装配站问题

```python

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
        a = [[7, 9, 3, 4, 8, 4], [8, 5, 6, 4, 5, 7]]
        t = [[2, 3, 1, 3, 4], [2, 1, 2, 2, 1]]
        e = [2, 4]
        x = [3, 2]
        n = 6
        self.fastway(a, t, e, x, n)
        >>> (38, 0)

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

```

### 矩阵链乘法顺序

```python

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

    def matrix_chain_multiply(self, A):
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

def __recursive_matrix_chain(self, p, m, s, i, j):
        '''
        矩阵链算法的低效递归版本
        '''
        if i == j:
            return 0
        m[i][j] = math.inf
        for k in range(i, j):
            q = self.__recursive_matrix_chain(p, m, s, i, k) + self.__recursive_matrix_chain(p, m, s, k + 1, j) + p[i] * p[k + 1] * p[j + 1] 
            if q < m[i][j]:
                m[i][j] = q
                s[i][j] = k + 1
        return m[i, j]
        
    def recursive_matrix_chain(self, p):
        '''
        矩阵链算法的低效递归版本
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        self.__recursive_matrix_chain(p, m, s, 0, n - 1)
        return (m, s)

    def memoized_matrix_chain(self, p):
        '''
        矩阵链算法的备忘录版本
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        # !备忘录版本与递归版本相同的地方都是要填表时进行递归，
        # !但是递归时并不重新计算表m中的元素,仅仅做一个某位置是否填过表的判断
        # 将表m全部填成无穷inf
        for i in range(n):
            for j in range(i, n):
                m[i][j] = math.inf
        self.loockup_chian(p, m, 0, n - 1)
        return m

    def loockup_chian(self, p, m, i, j):
        '''
        回溯查看表m中的元素
        '''
        # 查看而不是重新比较
        if m[i][j] < math.inf:
            return m[i][j]
        if i == j:
            m[i][j] = 0
        else:
            for k in range(i, j):
                q = self.loockup_chian(p, m, i, k) + \
                    self.loockup_chian(p, m, k + 1, j) + \
                    p[i] * p[k + 1] * p[j + 1] 
                if q < m[i][j]:
                    m[i][j] = q
        return m[i][j]

```

### 公共最长子序列

```python

def lcs_length(self, x, y):
        '''
        计算LCS的长度(也是矩阵路径的解法) 时间复杂度`O(mn)`

        Return
        ===
        (b ,c)
        '''
        m = len(x)
        n = len(y)
        c = zeros([m + 1, n + 1])
        b = zeros((m, n), dtype=np.str)
        for i in range(0, m):
            for j in range(0, n):
                if x[i] == y[j]:
                    c[i + 1][j + 1] = c[i][j] + 1
                    b[i][j] = '↖'
                elif c[i][j + 1] >= c[i + 1][j]:
                    c[i + 1][j + 1] = c[i][j + 1]
                    b[i][j] = '↑'
                else:
                    c[i + 1][j + 1] = c[i + 1][j]
                    b[i][j] = '←'
        return (c, b)

    def __lookup_lcs_length(self, x, y, c, b, i, j):
        if c[i][j] != math.inf:
            return c[i][j]
        if x[i - 1] == y[j - 1]:
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i - 1, j - 1) + 1
            b[i - 1][j - 1] = '↖'
        elif self.__lookup_lcs_length(x, y, c, b, i - 1, j) >= \
            self.__lookup_lcs_length(x, y, c, b, i, j - 1):
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i - 1, j)
            b[i - 1][j - 1] = '↑'
        else:
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i, j - 1)
            b[i - 1][j -1] = '←'
        return c[i][j]

    def memoized_lcs_length(self, x, y):
        '''
        公共子序列的备忘录版本 时间复杂度`O(mn)`
        '''
        m = len(x)
        n = len(y)
        c = zeros([m + 1, n + 1])
        b = zeros((m, n), dtype=np.str)
        #b = '↓'
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c[i][j] = math.inf
        for i in range(0, m):
            for j in range(0, n):
                b[i][j] = '↓'
        self.__lookup_lcs_length(x, y, c, b, m, n)
        return (c, b)

    def memoized_lcs_show(self, x, y):
        '''
        公共子序列的备忘录版本打印公共子序列 时间复杂度`O(mn)`
        '''
        c, b = self.memoized_lcs_length(x, y)
        print(c)
        print(b)
        self.print_lcs(b, x, len(x) - 1, len(y) - 1)
        print('')

    def print_lcs(self, b, X, i, j):
        '''
        打印公共子序列 运行时间为`O(m + n)`
        '''
        if i == -1 or j == -1:
            return
        if b[i ,j] == '↖':
            self.print_lcs(b, X, i - 1, j - 1)
            print(X[i], end=' ')
        elif b[i, j] == '↑':
            self.print_lcs(b, X, i - 1, j)
        else:
            self.print_lcs(b, X, i, j - 1)

    def print_lcs_with_tablec(self, c, X, Y, i, j):
        '''
        打印公共子序列 运行时间为`O(m + n)`
        '''
        if i == -2 or j == -2:
            return
        if c[i ,j] == c[i - 1][j - 1] + 1 and X[i] == Y[j]:
            self.print_lcs_with_tablec(c, X, Y, i - 1, j - 1)
            print(X[i], end=' ')
        elif c[i - 1, j] >= c[i][j - 1]:
            self.print_lcs_with_tablec(c, X, Y, i - 1, j)
        else:
            self.print_lcs_with_tablec(c, X, Y, i, j - 1)

```

### 最长递增子序列

```python

def longest_inc_seq(self, x):
        '''
        最长递增子序列(动态规划求解) `O(n^2)` 

        Example
        ===
        ```python
        >>> longest_inc_seq([2, 3, 1, 4])
        >>> [2, 3, 4]
        ```
        '''
        # 序列的长度
        n = len(x)
        # 动态规划子问题表的深度
        t = zeros([n, n])
        for i in range(n):
            for j in range(n):
                t[i][j] = math.inf
        last = 0
        max_count = 0
        max_count_index = 0
        seq = []
        for i in range(n):
            top = 0
            count = 1
            for j in range(i, n):
                if x[i] <= x[j] and top <= x[j]:
                    t[i][j] = x[j]
                    count += 1
                    top = x[j]
                    if count >= max_count:
                        max_count = count
                        max_count_index = i
                else:
                    t[i][j] = math.inf
        for i in range(n):
            val = t[max_count_index][i]
            if val != math.inf:
                seq.append(val)
        print(t)
        return seq

    def lower_bound(self, arr, x, start, end):
        middle = (start + end) // 2
        while arr[middle] < x:
            middle -= 1
        return middle

    def fast_longest_inc_seq(self, x):
        '''
        快速递归的最长递增子序列(二分查找) `O(nlgn)`
        '''
        n = len(x)
        g = []
        l = []
        # O(n)
        for i in range(n):
            g.append(math.inf)
        for i in range(n):
            # 二分查找 O(nlgn)
            k = self.lower_bound(g, x[i], 0, n -1)
            g[k] = x[i]
        # quick sort O(nlgn)
        g.sort()
        for i in range(n):
            if g[i] != math.inf:
                l.append(g[i])
        return l

```

### 最优二叉搜索树

#### 实现一 (非递归)

```python

   def optimal_bst(self, p, q, n):
        '''
        求最优二叉树
        '''
        e = zeros((n + 2, n + 1))
        w = zeros((n + 2, n + 1))
        root = zeros((n, n))
        for i in range(1, n + 2):
            e[i][i - 1] = q[i - 1]
            w[i][i - 1] = q[i - 1]
        for l in range(1, n + 1):
            for i in range(1, n - l + 1 + 1):
                j = i + l - 1
                e[i][j] = math.inf
                w[i][j] = w[i][j - 1] + p[j] + q[j]
                for r in range(i, j + 1):
                    t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                    if t < e[i][j]:
                        e[i][j] = t
                        root[i - 1][j - 1] = r
        e_return = zeros((n + 1, n + 1))
        w_return = zeros((n + 1, n + 1))
        for i in range(n):
            e_return[i] = e[i + 1]
            w_return[i] = w[i + 1]
        return (e_return, root)

```

#### 实现二 (递归+备忘录模式)

```python

    def __compute_weight(self, i : int, j : int, key : list, fkey : list, weight):
        if i - 1 == j:
            weight[i][j] = fkey[j]
        else:
            weight[i][j] = self.__compute_weight(i, j - 1, key, fkey, weight) + key[j] + fkey[j]
        return weight[i][j]
            
    def __dealbestBSTree(self, i : int, j : int, key : list, fkey : list, weight, min_weight_arr):
        '''
        备忘录模式(从上到下模式)
        '''
        if i - 1 == j:
            min_weight_arr[i][j] = weight[i][j]
            return weight[i][j]
        if min_weight_arr[i][j] != 0:
            return min_weight_arr[i][j]
        _min = 10
        for k in range(i, j + 1):
            tmp = self.__dealbestBSTree(i, k - 1, key, fkey, weight, min_weight_arr) + \
                self.__dealbestBSTree(k + 1, j, key, fkey, weight, min_weight_arr) + \
                weight[i][j]
            if tmp < _min:
                _min = tmp
        min_weight_arr[i][j] = _min
        return _min

    def bestBSTree(self, key : list, fkey : list):
        '''
        最优二叉搜索树的算法实现，这里首先采用自上而下的求解方法(动态规划+递归实现) `O(n^3)`
        '''
        n = len(key)
        min_weight_arr = zeros((n + 1, n))
        weight = zeros((n + 1, n))
        for k in range(1, n + 1):
            self.__compute_weight(k, n - 1, key, fkey, weight)
        self.__dealbestBSTree(1, n - 1, key, fkey, weight, min_weight_arr)
        m_w_r = zeros((n, n))
        w_r = zeros((n, n))
        for i in range(n):
            m_w_r[i] = min_weight_arr[i + 1]
            w_r[i] = weight[i + 1]
        return (w_r, m_w_r, min_weight_arr[1][n - 1]) 

    def show_bestBSTree(self, key : list, fkey : list):
        '''
        最优二叉搜索树的算法实现，这里首先采用自上而下的求解方法(动态规划+递归实现) `O(n^3)`
        并且打印出权重矩阵和最小权重
        '''
        w, m, min = self.bestBSTree(key, fkey)
        print('the weight matrix is')
        print(w)
        print('the min weight matrix is')
        print(m)
        print('the min weight value is')
        print(min)

```


[Github Code](https://github.com/Peefy/IntroductionToAlgorithm.Python/blob/master/src/chapter15)
