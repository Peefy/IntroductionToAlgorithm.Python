
# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
'''
Class Chapter8_1

Class Chapter8_2

Class Chapter8_3

Class Chapter8_4
'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter8_1:
    '''
    chpater8.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter8.1 note

        Example
        ====
        ```python
        Chapter8_1().note()
        ```
        '''
        print('chapter8.1 note as follow')
        print('第8章 线性时间排序')
        print('合并排序和堆排序在最坏情况下能达到O(nlgn),快速排序在平均情况下达到此上界面')
        print('第8章之前的排序算法都是比较排序')
        print('8.1节中将证明对含有n个元素的一个输入序列，', 
            '任何比较排序在最坏情况下都要用Ω(nlgn)次比较来进行排序')
        print('由此可知，合并排序和堆排序是最优的')
        print('本章还介绍三种线性时间排序,计数排序，基数排序，桶排序')
        print('8.1 排序算法运行时间的下界')
        print('决策树模型')
        print('比较排序可以被抽象地看作决策树，一颗决策树是一个满二叉树')
        print('在决策树中，每个节点都标有i：j，其中1<=i,j<=n,n是输入序列中元素的个数，控制结构，数据移动等都被忽略')
        print('如排序算法的决策树的执行对应于遍历从树的根到叶子节点的路径')
        print('要使排序算法能正确的工作，其必要条件是n个元素n！种排列中的每一种都要作为一个叶子出现')
        print('对于根结点来说，每一个叶子都可以是某条路径可以达到的')
        print('比较排序算法最坏情况下界，就是从根部到最底部叶子走过的最长路径，也就是树的高度nlgn')
        print('定理8.1 任意一个比较排序在最坏情况下，都需要做Ω(nlgn)次的比较')
        print('堆排序和合并排序都是渐进最优的比较排序算法,运行时间上界O(nlgn)')
        print('练习8.1-1 最小深度可能是n-1，对于已经n个排序好的元素比较n-1次即可，如三个元素比较两次')
        print('练习8.1-2 斯特林近似公式是求n！的一个近似公式')
        print('练习8.1-3 对于长度为n的n!种输入，至少一半而言，不存在线性运行时间的比较排序算法')
        print('练习8.1-4 现有n个元素需要排序，它包含n/k个子序列，每一个包含n个元素')
        print(' 每个子序列的所有元素都小于后续序列的所有元素，所以对n/k个子序列排序，就可以得到整个输入长度的排序')
        print(' 这个排序问题中所需的问题都需要有一个下界Θ(nlgk)')
        print('计数排序的基本思想是对每一个输入元素x，确定出小于x的元素个数，有了这一信息，', 
            '就可以把x直接放到最终输出数组中的位置上。例如有17个元素小于x，则x位于第18个位置上（元素互补相同）')
        print('在计数排序的代码中，假定输入是个数组A[1..n],length[A]=n')
        print('另外还需要两个数组，存放排序结果的B[1..n],以及提供临时存储区的C[0..k]')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_2:
    '''
    chpater8.2 note and function
    '''

    def countingsort2(self, A):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort2([0,1,1,3,4,6,5,3,5])
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        return self.countingsort(A, max(A) + 1)

    def countingsort(self, A, k):
        '''
        针对数组`A`计数排序，无需比较，非原地排序，当`k=O(n)`时，算法时间复杂度为`Θ(n)`,
        3个n for 循环
        需要预先知道数组元素都不大于`k`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B
    
    def note(self):
        '''
        Summary
        ====
        Print chapter8.2 note

        Example
        ====
        ```python
        Chapter8_2().note()
        ```
        '''
        print('chapter8.2 note as follow')
        print('8.2 计数排序')
        print('计数排序假设n个输入元素的每一个都是介于0到k之间的整数，', 
            '此处k为某个整数，k=O(n),计数排序的时间为O(n)')
        A = [5, 5, 4, 2, 1, 0, 3, 2, 1]
        print('数组A:', _deepcopy(A), '的计数排序：', self.countingsort(A, 6))
        print('计数排序虽然时间复杂度低并且算法稳定，但是空间复杂度高，并且需要先验知识`所有元素都不大于k`')
        print('计数排序的稳定性应用非常重要，而且经常作为基数排序的子程序，对于计数排序的正确性证明很重要')
        A = [6, 0, 2, 0, 1, 3, 4, 6, 1, 3, 2]
        print('练习8.2-1 数组A:', _deepcopy(A), '的计数排序：', self.countingsort(A, 7))
        A = [6, 0, 2, 0, 1, 3, 4, 6, 1, 3, 2]
        print(' 数组A:', _deepcopy(A), '另一种计数排序：', self.countingsort2(A))
        print('练习8.2-2 计数算法是稳定的')
        print('练习8.2-3 修改后算法不稳定，最好先放大数再放小数')
        print('练习8.2-4 略 不会')

        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_3:
    '''
    chpater8.3 note and function
    '''
    def getarraystr_subarray(self, A ,k):
        '''
        取一个数组中每个元素第k位构成的子数组

        Args
        ===
        `A` : 待取子数组的数组

        `k` : 第1位是最低位，第d位是最高位

        Return
        ===
        `subarray` : 取好的子数组

        Example 
        ===
        ```python
        Chapter8_3().getarraystr_subarray(['ABC', 'DEF', 'OPQ'], 1)
        ['C', 'F', 'Q']
        ```
        '''
        B = []
        length = len(A)
        for i in range(length):
            B.append(int(str(A[i])[-k]))
        return B
    
    def countingsort(self, A, k):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        k = 27
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B

    def radixsort(self, A, d):
        '''
        基数排序 平均时间复杂度为`Θ(nlgn)`

        Args
        ===
        `A` : 待排序的数组

        `d` : 数组A中每个元素都有d位数字/长度,其中第1位是最低位，第d位是最高位

        Return
        ===
        `sortedarray` : 排序好的数组 

        Example
        ===
        ```python
        >>> Chapter8_3().radixsort([54,43,32,21,11], 2)
        >>> [11, 21, 32, 43, 54]
        ```
        '''
        length = len(A)
        B = []
        for i in range(d):
            B.append(self.getarraystr_subarray(A, i + 1))
        for k in range(d):
            B[k] = self.countingsort(B[k], max(B[k]) + 1)
        C = _arange(length)
        for j in range(length):
            for i in range(d):            
                C[j] += B[i][j] * 10 ** i
            C[j] = C[j] - j
        return C 

    def note(self):
        '''
        Summary
        ====
        Print chapter8.3 note

        Example
        ====
        ```python
        Chapter8_3().note()
        ```
        '''
        print('chapter8.3 note as follow')
        print('8.3 基数排序')
        print('基数排序是用在老式穿卡机上的算法')
        print('关于这个算法就是按位排序要稳定')
        print('引理8.3 给定n个d位数，每一个数位有k个可能取值，', 
            '基数排序算法能够以Θ(d(n+k))的时间正确地对这些数排序')
        print('引理8.4 给定n个b位数，和任何正整数r<=b，', 
            'RADIX-SORT能以Θ((b/r)(n+2^r))的时间内正确地排序')
        print('基数排序的时间复杂度表达式中常数项比较大，',
            '若取b=O(lgn),r=lgn,则基数排序的时间复杂度为Θ(n)')
        A = ['ABC', 'DEF', 'OPQ']
        # print('数组A', _deepcopy(A), '的一个取子数组的样例：', self.getarraystr_subarray(A, 1))
        words = [54,43,32,21,11]
        print('练习8.3-1 数组words', _deepcopy(words), '的基数排序为:', self.radixsort(words, 2))
        print('练习8.3-2 排序算法的稳定性：假定在待排序的记录序列中，存在多个具有相同的关键字的记录，',
            '若经过排序，这些记录的相对次序保持不变，即在原序列中，r_i=r_j，且r_i在r_j之前，而在排序后的序列中，', 
            'r_i仍在r_j之前，则称这种排序算法是稳定的，否则成为不稳定的')
        print(' 常见排序算法的稳定性：堆排序，快速排序，希尔排序，直接选择排序不是稳定的排序算法，',
            '而基数排序，冒泡排序，直接插入排序，折半插入排序，合并排序时稳定的排序算法')
        print('练习8.3-3 证明过程中，计数排序要是稳定的才行，不然的话拆分子数组过程在复原后',
            '会导致相应的位数匹配不上，导致结果不正确')
        print('练习8.3-4 使用变形的计数排序，开辟n^2个空间')
        print('练习8.3-5 不会....看管d堆卡片吧')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_4:
    '''
    chpater8.4 note and function
    '''
    def insertsort(self, array):
        '''
        Summary
        ===
        插入排序的升序排列,时间复杂度`O(n^2)`
    
        Parameter
        ===
        `array` : a list like
        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> sort.insertsort(array)
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        A = array
        n = len(A)
        for j in range(1, n):
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
        return A

    def bucketsort(self, A):
        '''
        桶排序,期望时间复杂度`Θ(n)`(满足输入分布条件`[0,1)`的情况下)
        需要`链表list`额外的数据结构和存储空间

        Args
        ===
        `A` : 待排序的数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_4().bucketsort([0.5, 0.4, 0.3, 0.2, 0.1])
        >>> [0.1, 0.2, 0.3, 0.4, 0.5]
        ```
        '''
        n = len(A)
        B = []
        for i in range(n):
            B.insert(int(n * A[i]), A[i])
        return self.insertsort(B)
        
    def __find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''
        assert len(kettles1) == len(kettles2)
        n = len(kettles1)
        for i in range(n):
            for j in range(n):
                if kettles1[i] == kettles2[j]:
                    yield (i, j)

    def find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''      
        return list(self.__find_matching_kettle(kettles1, kettles2))

    def note(self):
        '''
        Summary
        ====
        Print chapter8.4 note

        Example
        ====
        ```python
        Chapter8_4().note()
        ```
        '''
        print('chapter8.4 note as follow')
        print('当桶排序的输入符合均匀分布时，即可以以线性时间运行，与计数排序类似，桶排序也对输入作了某种假设')
        print('具体来说，计数排序假设输入是由一个小范围内的整数构成，而桶排序则假设输入由一个随机过程产生，该过程将元素均匀地分布在区间[0,1)上')
        print('桶排序的思想就是把区间[0,1)划分成n个相同大小的子区间，或称桶。然后将n个输入分布到各个桶中去')
        print('因为输入数均匀分布在[0,1)上，所以一般不会有很多数落在一个桶中的情况')
        print('为了得到结果，先对桶中的数进行排序，然后按次序把桶中的元素列出来即可。')
        print('在桶排序的算法当中')
        print('桶排序算法中，假设输入的是一个含n个元素的数组A，且每个元素满足[0,1)')
        print('还需要一个辅助数组B[0..n-1]来存放链表(桶)，并假设可以用某种机制来维护这些表')
        print('桶排序的期望时间复杂度为Θ(n),证明过程略')
        A = [0.79, 0.13, 0.16, 0.64, 0.39, 0.20, 0.89, 0.53, 0.71, 0.42]
        print('练习8.4-1 数组A', _deepcopy(A), '的桶排序过程为:', self.bucketsort(A))
        print('练习8.4-2 略')
        print('练习8.4-3 E(X^2)=DX+E^2(X)=9/16+1 E^2(X)=E(X)E(X)=1')
        print('练习8.4-4 所有点到圆心的距离都服从均匀分布，所以采用桶排序')
        print('练习8.4-5 略')
        print('思考题8-1 给定n个不同的输入元素，对于任何确定或随机的比较排序算法，其期望运行时间都有下界Ω(nlgn)')
        print('思考题8-2 以线性时间原地置换排序:假设有一个由n个数据记录组成的数组要排序，且每个记录的关键字的值0或1')
        print(' 算法的运行时间为O(n),算法是稳定的,算法是原地排序的')
        print('思考题8-3 排序不同长度的数据项，字符串所有字符串字符的ascii码都不大于z，所以用基数排序，O(n)')
        print('思考题8-4 水壶：假设给定了n个红色的水壶和n个蓝色的水壶，他们的形状尺寸都不相同，所有红色水壶中所盛水的量都不同。')
        print(' 所有红色水壶中所盛水的量都不一样，蓝色水壶也是一样的；此外，对于每一个红色的水壶，都有一个对应的蓝色水壶，两者所盛的水量是一样的，反之亦然')
        print(' 任务是将匹配的红色水壶和蓝色水壶找出来，假设用1,2,3,4,5代表不同的水量')
        print('[1,2,3,4,5]和[5,4,3,2,1]的匹配为：', 
            self.find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        print(' 只会双重循环比较算法，对于下界为O(nlgn),考虑二叉树和递归算法吧,随机化算法也不会')
        print('思考题8-5 k排序，1排序就是完全排序')
        print(' 一个n元素的数组是k排序的，当且仅当对所有元素，当前A[i]<=A[i+k]')
        print('思考题8-6 合并已排序列表的下界')
        print(' 合并两个已知已排序列表这样的问题是经常出现的。它是合并排序的一个子过程')
        print(' 决策树说明比较次数有一个下界2n-o(n),还有一个更紧确的2n-1界')
        print('Han将排序算法的界改善至O(nlglgnlglglgn),尽管这些算法在理论上有重要的突破，', 
            '但都相当复杂，在目前来看，不太可能与现有的，正在实践中使用的排序算法竞争')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

chapter8_1 = Chapter8_1()
chapter8_2 = Chapter8_2()
chapter8_3 = Chapter8_3()
chapter8_4 = Chapter8_4()

def printchapter8note():
    '''
    print chapter8 note.
    '''
    print('Run main : single chapter eight!')  
    chapter8_1.note()
    chapter8_2.note()
    chapter8_3.note()
    chapter8_4.note()

# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
if __name__ == '__main__':  
    printchapter8note()
else:
    pass
