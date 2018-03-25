
# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
'''
Class Chapter9_1

Class Chapter9_2

Class Chapter9_3

Class Chapter9_4
'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

import pandas as pd

class Chapter9_1:
    '''
    chpater9.1 note and function
    '''

    def minimum(self, A : list) -> float:
        '''
        求集合中的最小值
        '''
        min = A[0]
        for i in range(1, len(A)):
            if min > A[i]:
                min = A[i]
        return min

    def note(self):
        '''
        Summary
        ====
        Print chapter9.1 note

        Example
        ====
        ```python
        Chapter9_1().note()
        ```
        '''
        print('chapter9.1 note as follow')
        print('第9章 中位数和顺序统计学')
        print('在一个由n个元素组成的集合中，第i个顺序统计量是该集合中第i小的元素。')
        print('非形式地说：一个中位数是它所在集合的\"中点元素\"')
        print('当n为奇数时，中位数是唯一的，出现在i=(n+1)/2处；')
        print('当n为偶数时，存在两个中位数，分别出现在i=n/2和i=n/2+1处')
        print('不考虑n的奇偶性，中位数总是出现在i=[(n+1)/2]处(下中位数)和i=[(n+1)/2]处(上中位数)')
        print('简单起见，书中的中位数总是指下中位数')
        print('本章讨论从一个由n个不同数值构成的集合中选择其第i个顺序统计量的问题，假设集合中的数互异')
        print('如下形式化地定义选择问题：')
        print(' 输入：一个包含n个(不同的)数的集合A和一个数i,1<=i<=n')
        print(' 输出：元素x属于A，它恰好大于A中其他i-1个数')
        print('选择问题可以在O(nlgn)时间内解决，因为可以用堆排序或合并排序对输入数据进行排序')
        print(' 然后在输出数组中标出第i个元素即可。但是还有其他更快的方法')
        print('9.1 最小值和最大值')
        A = [61, 52, 43, 34, 25, 16, 17]
        print('数组A', _deepcopy(A), '中元素的最小的元素为:', self.minimum(A))
        print('可以通过n-1次比较找出一个数组中的上界和下界')
        print('在某些应用中同时找出最小值和最大值')
        print('要设计出一个算法，使之通过渐进最优的Θ(n)次比较，能从n个元素中找出最小值和最大值')
        print('只要独立地找出最小值和最大值，各用n-1次比较，共有2n-2次比较')
        print('事实上，至多3[n/2]次比较久足以同时找出最小值和最大值，做法是记录比较过程中遇到的最大值和最小值')
        print('练习9.1-1 在最坏情况下，利用n+[lgn]-2次比较，即可找到n个元素中的第2小元素')
        print('练习9.1-2 在最坏情况下，同时找到n个数字中的最大值和最小值需要[3n/2]-2次比较')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_2:
    '''
    chpater9.2 note and function
    '''
    def partition(self, A : list, p : int, r : int) -> int:
        '''
        快速排序分堆子过程(并且避免了元素都相同时分堆进入最差情况)
        '''
        x = A[r]
        i = p - 1
        j = p - 1
        for j in range(p, r):
            if A[j] <= x:
                i = i + 1
                A[i], A[j] = A[j], A[i]
            if A[j] == x:
                j = j + 1
        A[i + 1], A[r] = A[r], A[i + 1]
        if j == r:
            return (p + r) // 2
        return i + 1

    def randomized_partition(self, A : list, p : int, r : int):
        '''
        快速排序随机分堆子过程
        '''
        i = _randint(p, r)
        A[r], A[i] = A[i], A[r]
        return self.partition(A, p, r)

    def __randomized_select(self, A : list, p : int, r : int, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`
        '''
        assert p < r      
        if len(A) == 0:
            return None
        if p == r:
            return A[p]
        q = self.randomized_partition(A, p, r)
        k = q - p + 1
        if i == k:
            return A[q]
        elif i < k:
            return self.__randomized_select(A, p, q - 1, i)
        return self.__randomized_select(A, q + 1, r, i - k)

    def randomized_select(self, A : list, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`,利用了`快速排序`分堆的方法(递归调用)
        '''
        assert i <= len(A) and i > 0
        return self.__randomized_select(A, 0, len(A) - 1, i)

    def randomized_select(self, A : list, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`,利用了`快速排序`分堆的方法(迭代调用)
        '''
        assert i <= len(A) and i > 0
        if len(A) == 0:
            return None
        return A[i - 1]

    def note(self):
        '''
        Summary
        ====
        Print chapter9.2 note

        Example
        ====
        ```python
        Chapter9_2().note()
        ```
        '''
        print('chapter9.2 note as follow')
        print('9.2 以期望线性时间做选择')
        print('一般选择问题看起来比找最小值的简单选择问题更难。但是，两种问题的渐进运行时间却是相同的：都是Θ(n)')
        print('将介绍一种用来解决选择问题的分治算法，Randomized-select算法，以排序算法为基本模型')
        print('如同排序在快速排序当中一样，此算法的思想也是对输入数组进行递归划分')
        print('但和快速排序不同的是，快速排序会递归处理划分的两边，而Randomized-select只处理划分的一边')
        print('所以快速排序的期望运行时间是Θ(n),而Randomized-select的期望运行时间为Θ(n),证明过程略')
        print('练习9.2-1 在Randomized-select中，对长度为0的数组，不会进行递归调用')
        print('练习9.2-2 指示器随机变量X_k和T(max(k-1,n-k))是独立的')
        print('练习9.2-3 略')
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第1小选择元素为：", self.randomized_select(A, 1))
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第2小选择元素为：", self.randomized_select(A, 2))
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第3小选择元素为：", self.randomized_select(A, 3))
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_3:
    '''
    chpater9.3 note and function
    '''
    def partition(self, A : list, p : int, r : int) -> int:
        '''
        快速排序分堆子过程(并且避免了元素都相同时分堆进入最差情况)
        '''
        x = A[r]
        i = p - 1
        j = p - 1
        for j in range(p, r):
            if A[j] <= x:
                i = i + 1
                A[i], A[j] = A[j], A[i]
            if A[j] == x:
                j = j + 1
        A[i + 1], A[r] = A[r], A[i + 1]
        if j == r:
            return (p + r) // 2
        return i + 1

    def select(self, A : list, i : int):
        '''
        在一个数组中选择出第i小的元素(当i=1时，即找出最小元素
        '''
        assert i <= len(A)
        return A[i]

    def note(self):
        '''
        Summary
        ====
        Print chapter9.3 note

        Example
        ====
        ```python
        Chapter9_3().note()
        ```
        '''
        print('chapter9.3 note as follow')
        print('9.3 最坏情况线性时间的选择')
        print('现在来看一个最坏情况运行时间为O(n)的选择算法Select')
        print('像9.2中的randomized_select一样，select通过对输入数组的递归划分来找出所求元素')
        print('但是，该算法的基本思想是要保证对数组的划分是个好的划分')
        print('select采用了取自快速排序的确定性划分算法patition并作出了一些修改，把划分主元元素作为其参数')
        print('算法SELECT通过执行下列步骤来确定一个有n>1个元素的输入数组中的第i小的元素。')
        print(' 1.将输入数组的n个元素划分为[n/5]组，每组5个元素，且至多只有一个组由剩下的n mod 5个元素组成')
        print(' 2.寻找[n/5]个组中每一组的中位数，首先对每组中的元素(至多为5个)进行插入排序，然后从排序过的序列中选出中位数')
        print(' 3.对第2步中找出的[n/5]个中位数，递归调用SELECT以找出其下中位数x')
        print(' 4.利用修改过的partition过程，按中位数的中位数x对输入数组进行划分。让k比划分低区的元素数目多1')
        print('  所以x是第k小的元素，并且有n-k个元素在划分的高区')
        print(' 5.如果i=k,则返回x。否则如果i<k,则在低区递归调用SELECT以找出第i小的元素，如果i>k,则在高区找第(i-k)个最小元素')
        print('因此，在[n/5]个组中，除了那个所包含元素可能少于5的组和包含x的那个组之外，至少有一半的组有3个元素大于x')
        print('类似地，小于x的原宿至少有3n/10-6个。因此，在最坏情况下，在第5步中最多有7n/10+6个元素递归调用select')
        print('步骤1,2,4需要O(n)的时间（步骤2对大小为O(1)的集合要调用O(n)次插入排序）')
        print('步骤3花时间T([n/5])，步骤5所需时间至多为T(7n/10+6),假设T是单调递增的')
        print('还需要做如下假设：即任何等于或少于140个元素的输入需要O(1)的时间；这个魔力常数140的起源很快就变得清晰了')
        print('在此假设下，可以得到递归式：')
        print('T(n)=Θ(1), n<=140; T(n)=T([n/5])+T(7n/10+6)+O(n), n>140)')
        print('用定义可以证明T(n)=O(n)')
        print('因此，select的最坏情况运行时间是线性的')
        print('与比较排序一样，select和randomized_select仅仅通过元素间的比较来确定它们之间的相对次序。')
        print('在第8章中，我们知道在比较模型中，即使是在平均情况下，排序仍然需要Ω(nlgn)')
        print('第8章的线性时间排序算法在输入上作了假设。相反地，本章的线性时间选择算法不需要关于输入的任何假设')
        print('它们不受下界Ω(nlgn)的约束，因为没有使用排序就解决了选择问题')
        print('所以本章中选择算法之所以具有线性运行时间，是因为这些算法没有进行排序；线性时间的行为并不是因为对输入做假设所得到的结果')
        print('第8章中的排序算法就是这么做的。在比较模型中，即使是在平均情况下，排序仍然需要Ω(nlgn)的时间')
        print('练习9.3-1 在算法select中，输入元素被分为每组5个元素')
        print(' 如果分成每组3个元素，select无法在线性时间内运行')
        print('练习9.3-2 证明如果n>=140,则至少有[n/4]个元素大于中位数的中位数x，并且至少有[n/4]个元素小于x')
        print('练习9.3-3 怎么让快速排序在最坏情况下以O(nlgn),随机化输入数组多次，使最坏情况发生的概率接近于0,但是运行时间常数项会增加')
        print('练习9.3-4 假设对一个包含有n个元素的集合，某算法只用比较来确定第i小的元素。')
        print(' 证明：无需另外的比较操作，它也能找到比i小的i-1个原宿和比i大的n-i个元素')
        print('练习9.3-5 假设已经有了一个用于求解中位数的黑箱子程序，它在最坏情况下需要线性运行时间。写出一个能解决任意顺序统计量的选择问题的线性时间算法')
        print('练习9.3-6 对于一个含有n个元素的集合来说，所谓k分位数，就是能把已排序的集合分成k个大小相等的集合的k-1个顺序统计量')
        print(' 给出一个能列出某一结合的k分位数的O(nlgk)时间的算法')
        print('练习9.3-7 给出一个O(n)时间的算法，在给定一个有n个不同数字的集合S以及一个正整数k<=n后，', 
            '它能确定出S中最接近其中位数的k个数')
        print('练习9.3-8 设X[1..n]和Y[1..n]为两个数组，每个都包含n个已经排好序的数。', 
            '给出一个求数组X和Y中所有2n个元素的中位数的，O(lgn)时间的算法')
        print('练习9.3-9 最短管道总长和问题')
        print('思考题9-1 已排序的i个最大数：给定一个含有n个元素的集合，希望能用一个基于比较的算法来找出按顺序排列的i个最大元素')
        print('思考题9-2 带权中位数：在O(nlgn)的最坏情况时间内求出n个元素的带权中位数')
        print('思考题9-3 小型顺序统计量:为从n个数字中选出第i个顺序统计量，',
            'SELECT在最坏情况下所使用的比较次数T(n)=Θ(n),但是常数项特别大')
        print('描述一个能用Ui(n)次比较找出n个元素中的第i小元素的算法，其中')
        print('Ui(n)=T(n) i>=n/2; Ui(n)=[n/2]+Ui([n/2])+T(2i)')
        print('证明：如果i<n/2,则Ui(n)=n+O(T(2i)lg(n/i))')
        print('证明：如果i是个小于n/2的常数，则Ui(n)=n+O(lgn)')
        print('证明：如果对k>=2有i=n/k,那么Ui(n)=n+O(T(2n/k)lgk)')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

chapter9_1 = Chapter9_1()
chapter9_2 = Chapter9_2()
chapter9_3 = Chapter9_3()

def printchapter9note():
    '''
    print chapter9 note.
    '''
    print('Run main : single chapter nine!')  
    chapter9_1.note()
    chapter9_2.note()
    chapter9_3.note()

# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
if __name__ == '__main__':  
    printchapter9note()
else:
    pass
