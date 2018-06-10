
## 概率分析和随机算法

### 逆序对的个数

```python

def __inversionListNum(self, array):

        # local function
        def __inversion(array, end):
            # 进行深拷贝保护变量
            list = _deepcopy([])
            n = _deepcopy(end)
            A = _deepcopy(array)
            if n > 1 :
                newList = __inversion(array, n - 1)
                # 相当于C#中的foreach(var x in newList); list.Append(x);
                for i in newList:
                    list.append(i)
            lastIndex = n - 1
            for i in range(lastIndex):
                if A[i] > A[lastIndex]:
                    list.append((i, lastIndex))
            return list

        return len(__inversion(array, len(array)))

```

### 随机打乱数组

```python

    def sortbykey(self, array, keys):
        '''
        根据keys的大小来排序array
        '''
        A = _deepcopy(array)
        length = len(A)
        for j in range(length):
            minIndex = j
            # 找出A中第j个到最后一个元素中的最小值
            # 仅需要在头n-1个元素上运行
            for i in range(j, length):
                if keys[i] <= keys[minIndex]:
                    minIndex = i
            # 最小元素和最前面的元素交换
            min = A[minIndex]
            A[minIndex] = A[j]
            A[j] = min
        return A

    def permute_bysorting(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_bysorting([1, 2, 3, 4])
        '''
        n = len(array)
        P = _deepcopy(array)
        for i in range(n):
            P[i] = _randint(1, n ** 3)
            _time.sleep(0.002)
        return self.sortbykey(array, P)

    def randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().randomize_inplace([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_without_identity(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_without_identity([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n - 1):
            _time.sleep(0.001)
            rand = _randint(i + 1, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_with_all(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_with_all([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            _time.sleep(0.001)
            rand = _randint(0, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array
    
    def permute_by_cyclic(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_by_cyclic([1, 2, 3, 4])
        '''
        A = _deepcopy(array)
        n = len(array)
        offset = _randint(0, n - 1)
        A = _deepcopy(array)
        for i in range(n):
            dest = i + offset
            if dest >= n:
                dest = dest - n
            A[dest] = array[i]
        return A

```
