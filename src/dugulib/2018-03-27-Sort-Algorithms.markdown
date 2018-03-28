---
layout: post
title: "Sorting Algorithms/排序算法"
description: "This is an introduction to sorting algorithms/这是一个排序算法的介绍"
categories: [Python]
tags: [python]
redirect_from:
  - /2018/03/27/
---

# Sorting Algorithms（排序算法）

## *Fisrt*

冒泡排序 `O(n^2)`     

鸡尾酒排序(双向冒泡排序) `O(n^2)`

插入排序 `O(n^2)`     

桶排序 `O(n)`         

计数排序 `O(n + k)`    

合并排序 `O(nlgn)`      

原地合并排序 `O(n^2)`   

二叉排序树排序 `O(nlgn)`  

鸽巢排序 `O(n+k)`

基数排序 `O(nk)`       

Gnome排序 `O(n^2)`

图书馆排序 `O(nlgn)`

## *Second*

选择排序 `O(n^2)`    

希尔排序 `O(nlgn)`   

组合排序 `O(nlgn)`

堆排序  `O(nlgn)`   

平滑排序  `O(nlgn)`

快速排序   `O(nlgn)`

Intro排序  `O(nlgn)`

Patience排序 `O(nlgn + k)`

## *Third*

Bogo排序 `O(n*n!)`

Stupid排序 `O(n^3)`

珠排序  `O(n) or O(sqrt(n))`

Pancake排序   `O(n)`

Stooge排序  `O(n^2.7)`   

### *冒泡排序* `O(n^2)`  

```python
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
```

### *插入排序* `O(n^2)`  

```python
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
```

### *选择排序* `O(n^2)`  

```python
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
```

### *合并排序* `O(nlgn)`  

```python
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
```

### *二叉堆排序* `O(nlgn)`  

```python
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

```

### *快速排序* 平均情况`O(nlgn)` 最差情况 `O(n^2)` 

```python
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

```

### *Stooge排序* `O(n^2.7)`

```python
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
```


[Github Code](https://github.com/Peefy/CLRS_dugu_code-master/tree/master/src/dugulib)
