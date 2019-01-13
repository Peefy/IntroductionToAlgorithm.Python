
## 几种排序算法

排序算法有很多，包括插入排序，冒泡排序，堆排序，归并排序，选择排序，计数排序，基数排序，桶排序，快速排序

## 插入排序
平均时间复杂度O(n^2), 最坏情况复杂度O(n^2), 最好情况复杂度O(n), 空间复杂度O(1), 原地排序，稳定算法

计算阶乘，多项式，逆序对

```python

    def insertSortAscending(self, array = []):
        '''
        Summary
        =
        插入排序的升序排列
        
        Parameter
        =
        array : a list like
        Return
        =
        sortedArray : 排序好的数组
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> Chapter2().insertSortAscending(array)
        >>> [1, 2, 3, 4, 5, 6]
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

    def insertSortDescending(self, array = []):
        '''
        Summary
        =
        插入排序的降序排列

        Parameter
        =
        array : a list like

        Return
        =
        sortedArray : 排序好的数组
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> Chapter2().insertSortAscending(array)
        >>> [6, 5, 4, 3, 2, 1]
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
            while(i >= 0 and A[i] < key):
                # 向右移动牌
                A[i + 1] = A[i]
                # 遍历之前的牌
                i = i - 1
            # 后来摸的牌插入相应的位置
            A[i + 1] = key
        # 输出降序排序后的牌
        return A

    def arrayContains(self, array = [], v = None):
        '''
        Summary
        =
        * a function
        * *检测一个数组中是否包含一个元素*

        Parameter
        =
        *array* : a list like
        v : a element

        Return
        =
        index:若找到返回找到的索引，没找到返回None

        Example:
        =
        >>> array = [12, 23, 34, 45]
        >>> v = 23
        >>> m = 55
        >>> Chapter2().arrayContains(array, v)
        >>> 1
        >>> Chapter2().arrayContains(array, m)
        >>> None
        '''
        index = None
        length = len(array)
        for i in range(length):
            if v == array[i]:
                index = i
        return index

    def twoNBinNumAdd(self, A = [], B = []):
        '''
        Summary
        =
        两个存放数组A和B中的n位二进制整数相加

        Parameter
        ====
        A : a list like and element of the list must be 0 or 1
        B : a list like and element of the list must be 0 or 1

        Return
        ======
        returnSum : sum of two numbers

        Example:
        =
        >>> A = [1, 1, 0, 0]
        >>> B = [1, 0, 0, 1]
        >>> Chapter2().twoNBinNumAdd(A, B)
        >>> [1, 0, 1, 0, 1]
        '''
        if len(A) != len(B):
            raise Exception('length of A must be equal to length of B')
        length = len(A)
        # 注意：range 函数和 arange 函数都是 左闭右开区间
        '''
        >>> range(0,3) 
        >>> [0, 1, 2]
        '''
        returnSum = arange(length + 1)
        bitC = 0
        for i in range(length):
            index = length - 1 - i
            bitSum = A[index] + B[index] + bitC
            if bitSum >= 2:
                bitSum = 0
                bitC = 1
            else:
                bitC = 0
            returnSum[index + 1] = bitSum
            if index == 0:
                returnSum[0] = bitC
        return returnSum

    def selectSortAscending(self, array = []):
        '''
        Summary
        =
        选择排序的升序排列
        
        Parameter
        =
        array : a list like
        Return
        =
        sortedArray : 排序好的数组
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> Chapter2().selectSortAscending(array)
        >>> [1, 2, 3, 4, 5, 6]
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

def insertSortWithIndex(self, array, start ,end):
        '''
        Summary
        ===
        插入排序的升序排列(带排序索引)
        
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

        Args
        ==
        array : 待排序的数组

        Returns
        ==
        sortedArray : 排序好的数组

        Example
        ==
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
    
    def calPolynomial(self, a_array, x):
        '''
        计算多项式

        Args
        ====
        a_array : 多项式系数的数组
        x : 待计算的多项式的代入未知数

        Return
        ======
        y : 计算的多项式的值

        Example
        =
        2x^2 + 2x + 1, x = 2, 2 * 2 ** 2 + 2 * 2 + 1 = 13
        >>> a_array = [2, 2, 1] 
        >>> Chapter2_3().hornerRule(a_array, 2)      
        >>> 13       
        '''
        n = len(a_array)
        total =0
        for i in range(n):
            total = total + a_array[i] * x ** (n - 1 - i)
        return total
  
    def calPolynomialWithHornerRule(self, a_array, x):
        '''
        用霍纳规则计算多项式

        Args
        ====
        a_array : 多项式系数的数组
        x : 待计算的多项式的代入未知数

        Return
        ======
        y : 计算的多项式的值

        Example
        =
        2x^2 + 2x + 1, x = 2, 2 * 2 ** 2 + 2 * 2 + 1 = 13
        >>> a_array = [2, 2, 1] 
        >>> Chapter2_3().hornerRule(a_array, 2)      
        >>> 13       
        '''
        y = 0
        n = len(a_array)
        for i in range(n):
            y = a_array[i] + x * y
        return y

    def __inversion(self, array, end):
        # 进行深拷贝保护变量
        list = deepcopy([])
        n = deepcopy(end)
        A = deepcopy(array)
        if n > 1 :
            newList = self.__inversion(array, n - 1)
            # 相当于C#中的foreach(var x in newList); list.Append(x);
            for i in newList:
                list.append(i)
        lastIndex = n - 1
        for i in range(lastIndex):
            if A[i] > A[lastIndex]:
                list.append((i, lastIndex))
        return list

    def inversion(self, array):
        '''
        递归方式求得数组中的所有逆序对，时间复杂度O(n * lg(n))

        Args
        =
        array : 代求逆序对的数组

        Return
        =
        list : 所有逆序对索引的集合，集合中的每一个元素都为逆序对的元组

        Example
        =
        >>> A = [2, 3, 8, 6, 1]
        >>> Chapter2_3().inversion(A)
        >>> [(0, 4), (1, 4), (2, 3), (2, 4), (3, 4)]      
        '''
        return self.__inversion(array, len(array))

    def inversionListNum(self, array):
        '''
        递归方式求得数组中的所有逆序对，时间复杂度O(n * lg(n))

        Args
        =
        array : 代求逆序对的数组

        Return
        =
        list_num : 所有逆序对索引的集合的长度

        Example
        =
        >>> A = [2, 3, 8, 6, 1]
        >>> Chapter2_3().inversionListNum(A)
        >>> 5     
        '''
        return len(self.inversion(array))

```
