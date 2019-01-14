
## 概率分析和随机算法

雇佣问题，应聘的代价与应聘者的等级顺序排列有关，全排列n!

指示器随机变量：离散随机变量X，事件A对应的指示器随机变量的期望值等于事件A发生的概率，伯努利分布。

所以雇佣问题第i者应聘者被录用的概率为E[Xi] = 1/i

所以雇佣的总人数为∑Exi = ∑1/i = lnn + O(1)

## 随机算法

很多时候无法知道有关输入的信息，一种做法是对输入进行控制，强加随机性

* *并不改变期望值* : 如雇佣程序和随机雇佣程序的期望雇佣次数仍然是lnn
* 依赖于随机的选择
* 对所有输入都是同样的期望值

n个元素任意排列出现的概率为1/n!即为随机全排列序列

## 生日悖论

一个人生日在某一天的概率为1/n,两个人生日相同的概率为1/n * 1/n * n= 1/n

一个房间里面的人数必须达到多少，才能使有两个人生日相同的概率达到一半，即所有人生日都不相同的概率小于一半即可,或者两个生日相同可选k个和(k-1)个两个人生日相同乘以1/n即可

## 球与盒子

把相同的球随机投到b个盒子里,在每个盒子都有球之前，要投多少个球，期望为blnb

几何分布投球的期望为1除以概率

在给定的盒子里面至少有一个球之前，平均至少要投b个球

落在给定盒子里面的球数服从几何分布，如果投n个球，落在给定盒子中的球数的期望值是n/b

## 投掷硬币序列

抛一枚均匀硬币n次，期望看到连续正面的最长序列有O(lgn)长

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
