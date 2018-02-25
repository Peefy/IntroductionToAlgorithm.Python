
# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py 

import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
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

    def insertSortWithIndex(self, array, start ,end):
        '''
        Summary
        =
        插入排序的升序排列(带排序索引)
        
        Parameter
        =
        array : a list like
        start : sort start index
        end : sort end index

        Return
        =
        sortedArray : 排序好的数组
        >>> array = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().insert(array, 1, 4)
        >>> [6 ,2, 3, 4, 5, 1]
        '''
        A = deepcopy(array)
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
        return A

    def __mergeSortOne2(self, array, p ,q, r):
        '''
        一步合并两堆牌排序算法过程

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
        # python中变量名和对象是分离的
        # 此时A是array的一个深拷贝，改变A不会改变array
        A = deepcopy(array)
        # 求数组的长度 然后分成两堆([p..q],[q+1..r]) ([0..q],[q+1..n-1])
        n = r + 1

        # 检测输入参数是否合理
        if q < 0 or q > n - 1:
            raise Exception("arg 'q' must not be in (0,len(array) range)")
        # n1 + n2 = n
        # 求两堆牌的长度
        n1 = q - p + 1
        n2 = r - q
        # 构造两堆牌(包含“哨兵牌”)
        L = arange(n1, dtype=float)
        R = arange(n2, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
        # 因为合并排序的前提是两堆牌是已经排序好的，所以这里排序一下
        # chapter2 = Chapter2()
        # L = chapter2.selectSortAscending(L)
        # R = chapter2.selectSortAscending(R)
        # 一直比较两堆牌的顶部大小大小放入新的堆中
        i, j = 0, 0
        for k in range(p, n):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
            # 如果L牌堆放完了
            if i == n1:
                # R牌堆剩下的牌数量
                remainCount = n2 - j
                # 把R牌堆剩下的按顺序放到新的牌堆中
                for m in range(remainCount):
                    k += 1
                    A[k] = R[j]
                    j += 1
                break
            # 如果R牌堆放完了
            if i == n2:
                # L牌堆剩下的牌数量
                remainCount = n1 - i
                # 把L牌堆剩下的按顺序放到新的牌堆中
                for m in range(remainCount):
                    k += 1
                    A[k] = L[i]
                    i += 1
                break
        return A

    def __mergeSortOne(self, array, p ,q, r):
        '''
        一步合并两堆牌排序算法过程

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
        # python中变量名和对象是分离的
        # 此时A是array的一个引用
        A = array
        # 求数组的长度 然后分成两堆([p..q],[q+1..r]) ([0..q],[q+1..n-1])
        n = r + 1

        # 检测输入参数是否合理
        if q < 0 or q > n - 1:
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
        # 加入无穷大“哨兵牌”, 对不均匀分堆的完美解决
        L[n1] = math.inf
        R[n2] = math.inf
        # 因为合并排序的前提是两堆牌是已经排序好的，所以这里排序一下
        # chapter2 = Chapter2()
        # L = chapter2.selectSortAscending(L)
        # R = chapter2.selectSortAscending(R)
        # 一直比较两堆牌的顶部大小大小放入新的堆中
        i, j = 0, 0
        for k in range(p, n):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
        return A

    def __mergeSort(self, array, start, end):
        '''
        合并排序总过程

        Args:
        =
        array : 待排序数组
        start : 排序起始索引
        end : 排序结束索引

        Return:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().mergeSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        '''
        # python一切皆对象和引用，所以要拷贝...特别是递归调用的时候
        r = deepcopy(end)
        p = deepcopy(start)
        if p < r:
            # 待排序序列劈成两半
            middle = int((r + p) / 2)
            q = deepcopy(middle)
            # 递归调用
            # array =  self.__mergeSort(array, start, middle)
            self.__mergeSort(array, p, q)
            # 递归调用
            # array = self.__mergeSort(array, middle + 1, end)
            self.__mergeSort(array, q + 1, r)
            # 劈成的两半牌合并
            # array = self.__mergeSortOne(array, start ,middle, end)
            self.__mergeSortOne(array, p, q, r)
        return array    

    def mergeSort(self, array):
        '''
        归并排序：最优排序复杂度n * O(log2(n)), 空间复杂度O(n)

        Args:
        =
        array : 待排序的数组

        Returns:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().mergeSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__mergeSort(array, 0, len(array) - 1)

    def __mergeSortWithSmallArrayInsertSort(self, array, start, end, k):
        p = deepcopy(start)
        r = deepcopy(end)        
        if r - p + 1 > k:
            # 待排序序列劈成两半
            middle = int((r + p) / 2)
            q = deepcopy(middle)
            self.__mergeSortWithSmallArrayInsertSort(array, p, q, k)
            self.__mergeSortWithSmallArrayInsertSort(array, q + 1, r, k)
            self.__mergeSortOne(array, p, q, r)
        return self.insertSortWithIndex(array, p, r)

    def mergeSortWithSmallArrayInsertSort(self, array, k = 4):
        '''
        合并排序 ： 将待排序数组拆分到一定程度(而不是单个元素数组时),子问题足够小时采用插入排序

        Args:
        =
        array : 待排序的数组
        k : 子问题采用插入排序的最小规模

        Returns:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().mergeSortWithSmallArrayInsertSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__mergeSortWithSmallArrayInsertSort(array, 0, len(array) - 1, k)

    def __insertSort(self, array, num):
        key = array[num]
        # 反向查找
        for i in range(num):
            index = num - i - 1
            # 右移           
            if(key <= array[index]):
                array[index + 1] = array[index]
            # 插入
            else:
                array[index + 1] = key
                break
        return array

    def insertSort(self, array, num):
        '''
        递归版本的插入排序

        Args:
        ====
        array : 待排序的数组

        Return: 
        ======
        sortedArray : 排序好的数组

        Example: 
        =
        >>> Chapter2_3().insertSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        '''
        if num > 0:
            # O(1)
            self.insertSort(array, num - 1)  
            # O(n)
            self.__insertSort(array, num)   
        return array

    def factorial(self, n):
        '''
        Factorial of n

        Args:
        =
        n : the factorial number

        Return:
        =
        factorial : the factorial of the number

        Example:
        =
        >>> Chapter2_3().factorial(4)
        >>> 24

        '''
        # Don't forget to terminate the iterations
        if n <= 0:
            return 1
        return n * self.factorial(n - 1)

    def insertSortDichotomy(self, array, index):
        '''
        二分法插入排序

        Args:
        =
        array : 待排序的数组

        Return:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().insertSortDichotomy([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        A = deepcopy(array)
        n = len(A)
        if index >= n or index < 0 : 
            raise Exception("arg 'index' must be in range [0,len(array))")
        for j in range(1, index + 1):
            ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
            # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
            # 后来摸的牌
            key = A[j]
            # 之前手中的已经排序好的牌的最大索引
            i = j - 1
            low = 0
            high = i
            insertIndex = 0
            # 二分法寻找插入的位置
            while low <= high :
                middle = int((low + high) / 2)
                if key >= A[middle] and key <= A[middle + 1] :
                    insertIndex = middle + 1
                if key > A[middle]:
                    high = middle - 1
                if key < A[middle]:
                    low = middle + 1
            # 移动牌
            while(i >= insertIndex):
                # 向右移动牌
                A[i + 1] = A[i]
                # 遍历之前的牌
                i = i - 1
            # 后来摸的牌插入相应的位置
            A[i + 1] = key
        # 输出升序排序后的牌
        return A
        
    def __sumOfTwoNumbersEqual(self, array ,lastIndex, x):
        n = len(array)
        for i in range(0, lastIndex):
            if abs(array[i] + array[lastIndex] - x) < 10e-5:
                return True
        return False 

    def __internalSumOfTwoNumbersEqual(self, array, index, x):
        isFind = False
        n = deepcopy(index)
        # 如果n<0就结束递归
        if n < 0:
            return
        A = deepcopy(array)
        # 如果存在就结束递归
        result = self.__internalSumOfTwoNumbersEqual(A, n - 1, x)
        if result == True:
            return True
        return self.__sumOfTwoNumbersEqual(A, n, x) 

    def sumOfTwoNumbersEqual(self, array, x):
        '''
        判断出array中是否存在有两个其和等于x的元素

        Args:
        =
        array : 待判断的集合
        x : 待判断的元素

        Return:
        =
        result -> bool : 是否存在

        Example:
        =
        >>> A = [1, 2, 3]
        >>> Chapter2_3().sumOfTwoNumbersEqual(A, 3)
        >>> True
        >>> A = [1, 2, 3, 4]
        >>> Chapter2_3().sumOfTwoNumbersEqual(A, 9)
        >>> False
        '''
        return self.__internalSumOfTwoNumbersEqual(array, len(array) - 1, x)

    def __bubbleSort(self, array, start, end):
        A = deepcopy(array)
        p = deepcopy(start)
        q = deepcopy(end)
        if p > q:
            raise Exception('The start index must be less than the end index')
        length = q + 1
        for i in range(p, length):
            for j in range(i + 1, length):
                if A[j] < A[j - 1]:
                    # 禁止python的方便写法：A[j], A[j - 1] = A[j - 1], A[j]
                    # temp = A[j]
                    # A[j] = A[j - 1]
                    # A[j - 1] = temp
                    A[j], A[j - 1] = A[j - 1], A[j]
        return A

    def bubbleSort(self, array):
        '''
        冒泡排序，时间复杂度o(n ** 2)

        Args
        ====
        array : 排序前的数组

        Return
        ======
        sortedArray : 使用冒泡排序排好的数组

        Example:
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().bubbleSort(A)
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__bubbleSort(array, 0, len(array) - 1)

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
        A = [2.1, 12.2, 45.6, 12, 36.2, 50]
        print('单步合并排序前的待排序数组', A)
        print('单步合并排序后的数组(均匀分堆)', self.__mergeSortOne(A, 0, 2, len(A) - 1))
        B = [23, 45, 67, 12, 24, 35, 42, 54]
        print('单步合并排序前的待排序数组', B)
        print('单步合并排序后的数组(非均匀分堆)', self.__mergeSortOne(B, 0, 2, len(B) - 1))
        print('单步合并排序在两堆牌已经是有序的条件下时间复杂度是O(n),因为不包含双重for循环')
        A = [6, 5, 4, 3, 2, 1]
        # 浅拷贝
        A_copy = copy(A)
        print('总体合并算法应用，排序', A_copy, '结果:', self.mergeSort(A))
        print('2.3.2 分治法分析')
        print(' 当一个算法中含有对其自身的递归调用时，其运行时间可用一个递归方程(递归式)来表示')
        print(' 如果问题的规模足够小，则得到直接解的时间为O(1)')
        print(' 把原问题分解成a个子问题，每一个问题的大小是原问题的1/b，分解该问题和合并解的时间为D(n)和C(n)')
        print(' 递归式：T(n) = aT(n/b) + D(n) + C(n)')
        print('合并排序算法分析')
        print('合并算法在元素是奇数个时仍然可以正确地工作，假定原问题的规模是2的幂次')
        print('当排序规模为1时，需要常量时间，D(n) = O(1)')
        print(' 分解：这一步仅仅是计算出子数组的中间位置，需要常量时间，所以D(n) = O(1)')
        print(' 解决：递归地解两个规模为n/2的子问题，时间为2T(n/2)')
        print(' 合并：在含有n个元素的子数组上，单步分解合并的时间C(n) = O(n)')
        print(' 合并排序算法的递归表达式T(n) = O(1) n = 1;2T(n/2) + O(n)')
        print('最后解得T(n) = O(n * log2(n))')
        print('练习2.3-1：将输入数组A分解为八个数，然后从底层做合并排序，3和41合并为3,41；52和26合并为26,52，再合并为3,26,41,52,后四个数同理，然后做总的合并')
        A = [3, 41, 52, 26, 38, 57, 9, 49]
        print('数组A=[3,41,52,26,38,57,9,49]的合并排序结果为：', self.mergeSort(A))
        print('练习2.3-2：(tip:哨兵牌可以的)')
        A = [3, 41, 52, 16, 38, 57, 79, 99]
        print('数组A=[3, 41, 52, 16, 38, 57, 79, 99]的单步分解合并排序结果为：', self.__mergeSortOne2(A, 0, 2, len(A) - 1))
        print('练习2.3-3：每一步的递推公式 + 上一步的递推公式左右两边*2即可得通式T(n) = nlog2(n)')
        print('练习2.3-4: 递归插入方法')
        A = [1, 2, 3, 5, 6, 4]
        print('数组A=[1, 2, 3, 5, 6, 4]的递归单步插入排序结果为：', self.__insertSort(A, len(A) - 1))
        A = [3, 41, 12, 56, 68, 27, 19, 29]
        print('数组A=[3, 41, 12, 56, 68, 27, 19, 29]的递归插入排序结果为：', self.insertSort(A, len(A) - 1))
        print('递归插入的T(1) = O(1) ; T(n) = T(n-1) + n')
        print('练习2.3-6:二分法插入排序')
        # 插入排序最坏情况
        A = [6, 5, 4, 3, 2, 1]
        print('数组A=[1, 2, 3, 5, 6, 4]的二分法插入排序结果为：', self.insertSortDichotomy(A, len(A) - 1))
        print('阶乘的递归', self.factorial(4))
        print('练习2.3-7(鉴戒并归排序的思路或者阶乘的思路，递归)')
        print('[6,5,4,3,2,1]中找5的结果是：', self.sumOfTwoNumbersEqual(A, 5))
        print('[6,5,4,3,2,1]中找11的结果是：', self.sumOfTwoNumbersEqual(A, 11))
        print('[6,5,4,3,2,1]中找12的结果是：', self.sumOfTwoNumbersEqual(A, 12))
        print('思考题2-1:在并归排序中对小数组采用插入排序')
        print(' 带索引的插入排序如下：')
        print(' [6,5,4,3,2,1]从索引1到4的排序为：', self.insertSortWithIndex(A, 1, 4))
        A = [8, 7, 6, 5, 4, 3, 2, 1]
        print(' [8,7,6,5,4,3,2,1]从小问题采用插入排序的合并排序结果为：', 
            self.mergeSortWithSmallArrayInsertSort(A))
        print(' [8,7,6,5,4,3,2,1]从小问题采用插入排序的合并排序结果为(最小规模为3)：', 
            self.mergeSortWithSmallArrayInsertSort(A, 3))
        print(' 1.最坏的情况下，n/k个子列表(每个子列表的长度为k)可以用插入排序在O(nk)时间内完成排序')
        print(' 2.这些子列表可以在O(nlg(n/k)最坏情况时间内完成合并)')
        print(' 3.修改后的合并排序算法的最坏情况运行时间为O(nk+nlg(n/k)),k的最大渐进值为1/n')
        print(' 4.在实践中，k的值应该按实际情况选取')
        print('思考题2-2:冒泡排序的正确性')
        print(' [8,7,6,5,4,3,2,1]的冒泡排序结果为：', 
            self.bubbleSort(A))
        print('思考题2-3:用于计算多项式的霍纳规则的正确性')
        print('思考题2-4:逆序对')
        # python src/chapter2/chapter2_3.py
        # python3 src/chapter2/chapter2_3.py

if __name__ == '__main__':
    Chapter2_3().note()
else:
    pass

# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py

