
## 平摊分析

在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的

平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能

平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法

```python

def multipop(self, S : list, k):
        '''
        栈的弹出多个数据的操作
        '''
        while len(S) > 0 and k > 0:
            S.pop()
            k = k - 1

def increment(self, A : list):
        i = 0
        while i < len(A) and A[i] == 1:
            A[i] = 0
            i += 1
        if i < len(A):
            A[i] = 1

```