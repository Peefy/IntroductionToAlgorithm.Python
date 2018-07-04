
## 线性时间排序

```python
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

```

