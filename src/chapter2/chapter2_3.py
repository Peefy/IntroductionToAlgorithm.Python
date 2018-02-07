
# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py 
import sys
import math
from numpy import arange
from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

if __name__ == '__main__':
    from chapter2 import Chapter2
else:
    from .chapter2 import Chapter2
    
class Chapter2_3:
    '''
    CLRS 第二章 2.3
    '''
    def mergeSort(self, array, q):
        '''
        合并排序算法过程

        Args
        =
        array : a array like

        Returns:
        =
        sortedArray : 排序好的数组

        Raises:
        =
        None
        '''
        A = array
        # 求数组的长度 然后分成两堆([p..q],[q+1..r]) ([0..q],[q+1..n-1])
        n = len(A)
        # 将p和r替换为0和n-1
        p = 0
        r = n - 1
        # 检测输入参数是否合理
        if q <= 0 or q >= n - 1:
            raise Exception("arg 'q' must not be in (0,len(array) range)")
        # n1 + n2 = n
        # 求两堆牌的长度
        n1 = q - p + 1
        n2 = r - q
        # 构造两堆牌(包含“哨兵牌”)
        L = arange(n1 + 1, dtype=float)
        R = arange(n2 + 1, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
        # 加入“哨兵牌”
        L[n1] = math.inf
        R[n2] = math.inf
        # 因为合并排序的前提是两堆牌是已经排序好的，所以这里排序一下
        chapter2 = Chapter2()
        L = chapter2.selectSortAscending(L)
        R = chapter2.selectSortAscending(R)
        # 比较大小放入新的堆中
        i, j = 0, 0
        for k in range(p, r):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
        return A

    def note(self):
        '''
        Summary
        =
        Print chapter2.3 note
        Example
        =
        >>> Chapter2_3().note()
        '''
        print('chapter 2.3 note')
        print('算法设计有很多方法')
        print('如插入排序方法使用的是增量方法，在排好子数组A[1..j-1]后，将元素A[j]插入，形成排序好的子数组A[1..j]')
        print('2.3.1 分治法')
        print(' 很多算法在结构上是递归的，所以采用分治策略，将原问题划分成n个规模较小而结构与原问题相似的子问题，递归地解决这些子问题，然后再合并其结果，就得到原问题的解')
        print(' 分治模式在每一层递归上都有三个步骤：分解，解决，合并')
        print(' 分治法的一个例子:合并排序')
        print('  分解：将n个元素分成各含n/2个元素的子序列；')
        print('  解决：用合并排序法对两个子序列递归地排序；')
        print('  合并：合并两个已排序的子序列以得到排序结果')
        print(' 在对子序列排序时，其长度为1时递归结束。单个元素被视为是已经排序好的')
        print(' 合并排序的关键步骤在与合并步骤中的合并两个已排序子序列')
        print(' 引入辅助过程MERGE(A,p,q,r),A是个数组，p,q,r是下标,满足p<=q<r,将数组A拆分成两个子数组A[p,q]和A[q+1,r]')
        print(' 数组A的长度为n = r - p + 1，合并过程的时间代价为O(n)')
        print(' 用扑克牌类比合并排序过程，假设有两堆牌都已经排序好，牌面朝上且最小的牌在最上面，期望结果是这两堆牌合并成一个排序好的输出堆，牌面朝下放在桌上')
        print(' 步骤是从两堆牌的顶部的两张牌取出其中较小的一张放在新的堆中，循环这个步骤一直到两堆牌中的其中一堆空了为止，再将剩下所有的牌放到堆上即可')
        # 合并排序针对的是两堆已经排序好的两堆牌，这样时间复杂度为O(n)
        A = [12.1, 2.2, 45.6, 32, 56.2, 10]
        print('合并排序前的待排序数组', A)
        print('合并排序后的数组', self.mergeSort(A, 2))

        # python src/chapter2/chapter2_3.py
        # python3 src/chapter2/chapter2_3.py
        print('')

if __name__ == '__main__':
    Chapter2_3().note()
else:
    pass

# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py

