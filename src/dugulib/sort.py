
'''
排序算法集合

First
=====

冒泡排序 `O(n^2)`      ok

鸡尾酒排序(双向冒泡排序) `O(n^2)`

插入排序 `O(n^2)`      ok

桶排序 `O(n)`          ok

计数排序 `O(n + k)`     ok

合并排序 `O(nlgn)`      ok

原地合并排序 `O(n^2)`    ok

二叉排序树排序 `O(nlgn)`  ok

鸽巢排序 `O(n+k)`

基数排序 `O(nk)`        ok

Gnome排序 `O(n^2)`

图书馆排序 `O(nlgn)`

Second
======

选择排序 `O(n^2)`    ok

希尔排序 `O(nlgn)`   

组合排序 `O(nlgn)`

堆排序  `O(nlgn)`   ok

平滑排序  `O(nlgn)`

快速排序   `O(nlgn)`

Intro排序  `O(nlgn)`

Patience排序 `O(nlgn + k)`

Third
=====

Bogo排序 `O(n*n!)`

Stupid排序 `O(n^3)`

珠排序  `O(n) or O(sqrt(n))`

Pancake排序   `O(n)`

Stooge排序  `O(n^2.7)`   ok

'''

# python src/dugulib/sort.py
# python3 src/dugulib/sort.py
from __future__ import division, absolute_import, print_function
import math as _math
import random as _random
from copy import deepcopy as _deepcopy
from numpy import arange as _arange

__all__ = ['insertsort', 'selectsort', 'bubblesort',
               'mergesort', 'heapsort', 'quicksort', 
               'stoogesort'].sort()

class Sort:
    '''
    排序算法集合类
    '''
    def insertsort(self, array : list) -> list:
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

    def selectsort(self, array : list = []) -> list:
        '''
        Summary
        ===
        选择排序的升序排列,时间复杂度:O(n^2):
        
        Args
        ===
        `array` : a list like

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> sort.selectsort(array)
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        A = array
        length = len(A)
        for j in range(length):
            minIndex = j
            # 找出A中第j个到最后一个元素中的最小值
            # 仅需要在头n-1个元素上运行
            for i in range(j, length):
                if A[i] <= A[minIndex]:
                    minIndex = i
            # 最小元素和最前面的元素交换
            min = A[minIndex]
            A[minIndex] = A[j]
            A[j] = min
        return A

    def bubblesort(self, array : list) -> list:
        '''
        冒泡排序,时间复杂度`O(n^2)`

        Args
        ====
        `array` : 排序前的数组

        Return
        ======
        `sortedArray` : 使用冒泡排序排好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> sort.bubblesort(A)
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        nums = _deepcopy(array)
        for i in range(len(nums) - 1):    
            for j in range(len(nums) - i - 1):  
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums

    def __mergeSortOne(self, array : list, p : int ,q : int, r : int) -> list:
        '''
        一步合并两堆牌排序算法过程

        Args
        ===
        `array` : a array like

        Returns
        ===
        `sortedArray` : 排序好的数组

        Raises
        ===
        `None`
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
        L = _arange(n1 + 1, dtype=float)
        R = _arange(n2 + 1, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
        # 加入无穷大“哨兵牌”, 对不均匀分堆的完美解决
        L[n1] = _math.inf
        R[n2] = _math.inf
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

    def __mergeSort(self, array : list, start : int, end : int) -> list:
        '''
        合并排序总过程

        Args
        ===
        `array` : 待排序数组
        `start` : 排序起始索引
        `end` : 排序结束索引

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> sort.mergeSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        # python一切皆对象和引用，所以要拷贝...特别是递归调用的时候
        r = _deepcopy(end)
        p = _deepcopy(start)
        if p < r:
            # 待排序序列劈成两半
            middle = (r + p) // 2
            q = _deepcopy(middle)
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

    def mergesort(self, array : list) -> list:
        '''
        归并排序/合并排序：最优排序复杂度`O(n * log2(n))`, 空间复杂度`O(n)`

        Args
        ===
        array : 待排序的数组

        Returns
        ===
        sortedArray : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> sort.mergesort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        return self.__mergeSort(array, 0, len(array) - 1)

    def left(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:左儿子:的下标
        '''
        return int(2 * i + 1)

    def right(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:右儿子:的下标
        '''
        return int(2 * i + 2)

    def parent(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:父节点:的下标
        '''
        return (i + 1) // 2 - 1

    def heapsize(self, A : list) -> int:
        '''
        求一个数组形式的:二叉堆:的:堆大小:
        '''
        return len(A) - 1

    def maxheapify(self, A : list, i : int) -> list:
        '''
        保持堆使某一个结点i成为最大堆(其子树本身已经为最大堆) :不使用递归算法:
        '''
        count = len(A)
        largest = count
        while largest != i:
            l = self.left(i)
            r = self.right(i)
            if l <= self.heapsize(A) and A[l] >= A[i]:
                largest = l
            else:
                largest = i
            if r <= self.heapsize(A) and A[r] >= A[largest]:
                largest = r
            if largest != i:
                A[i], A[largest] = A[largest], A[i]
                i, largest = largest, count
        return A

    def buildmaxheap(self, A : list) -> list:
        '''
        对一个数组建立最大堆的过程, 时间代价为:O(n):
        '''
        count = int(len(A) // 2)
        for i in range(count + 1):
            self.maxheapify(A, count - i)
        return A

    def heapsort(self, A : list) -> list:
        '''
        堆排序算法过程, 时间代价为:O(nlgn):

        Args
        ===
        A : 待排序的数组A

        Return
        ====
        sortedA : 排序好的数组

        Example
        ====
        ```python
        >>> import sort
        >>> sort.heapsort([7, 6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6, 7]
        ```
        '''
        heapsize = len(A) - 1

        def left(i : int):
            '''
            求:二叉堆:一个下标i的:左儿子:的下标
            '''
            return int(2 * i + 1)

        def right(i : int):
            '''
            求:二叉堆:一个下标i的:右儿子:的下标
            '''
            return int(2 * i + 2)

        def parent(i : int):
            '''
            求:二叉堆:一个下标i的:父节点:的下标
            '''
            return (i + 1) // 2 - 1

        def __maxheapify(A : list, i : int):
            count = len(A)
            largest = count
            while largest != i:
                l = left(i)
                r = right(i)
                if  l <= heapsize and A[l] >= A[i]:
                    largest = l
                else:
                    largest = i
                if r <= heapsize and A[r] >= A[largest]:
                    largest = r
                if largest != i:
                    A[i], A[largest] = A[largest], A[i]
                    i, largest = largest, count
            return A

        self.buildmaxheap(A)
        length = len(A)   
        for i in range(length - 1):
            j = length - 1 - i
            A[0], A[j] = A[j], A[0]
            heapsize = heapsize - 1
            __maxheapify(A, 0)
        return A

    def partition(self, A : list, p : int, r : int):
        '''
        快速排序的数组划分子程序
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

    def __quicksort(self, A : list, p : int, r : int):
        left = _deepcopy(p)
        right = _deepcopy(r)
        if left < right:
            middle = _deepcopy(self.partition(A, left, right))
            self.__quicksort(A, left, middle - 1)
            self.__quicksort(A, middle + 1, right)

    def quicksort(self, A : list):
        '''
        快速排序，时间复杂度`o(n^2)`,但是期望的平均时间较好`Θ(nlgn)`

        Args
        ====
        `A` : 排序前的数组`(本地排序)`

        Return
        ======
        `A` : 使用快速排序排好的数组`(本地排序)`

        Example
        ===
        ```python
        >>> import sort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> sort.quicksort(A)
        >>> [1, 2, 3, 4, 5, 6]
        '''
        self.__quicksort(A, 0, len(A) - 1)
        return A

    def __stoogesort(self, A, i, j):
        if A[i] > A[j]:
            A[i], A[j] = A[j], A[i]
        if i + 1 >= j:
            return A
        k = (j - i + 1) // 3
        __stoogesort(A, i, j - k)
        __stoogesort(A, i + k, j)
        return __stoogesort(A, i, j - k)

    def stoogesort(self, A : list) -> list:
        '''
        Stooge原地排序 时间复杂度为:O(n^2.7):

        Args
        ===
        `A` : 排序前的数组:(本地排序):
        '''
        return __stoogesort(A, 0, len(A) - 1)

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
        >>> countingsort2([0,1,1,3,4,6,5,3,5])
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
        >>> countingsort([0,1,1,3,4,6,5,3,5], 6)
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

    def getarraystr_subarray(self, A, k):
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
        getarraystr_subarray(['ABC', 'DEF', 'OPQ'], 1)
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
        >>> countingsort([0,1,1,3,4,6,5,3,5], 6)
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

_inst = Sort()
insertsort = _inst.insertsort
selectsort = _inst.selectsort
bubblesort = _inst.bubblesort
mergesort = _inst.mergesort
heapsort = _inst.heapsort
quicksort = _inst.quicksort
stoogesort = _inst.stoogesort

def test():
    '''
    sort.insertsort test

    sort.selectsort test

    sort.bubblesort test

    sort.mergesort test

    sort.heapsort test

    sort.quicksort test
    '''
    print(insertsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(selectsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(bubblesort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(mergesort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(heapsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(quicksort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print('module sort test successful!!')

if __name__ == '__main__':
    # python src/dugulib/sort.py
    # python3 src/dugulib/sort.py
    test()
else:
    pass

