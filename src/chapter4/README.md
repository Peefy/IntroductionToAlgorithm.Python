
## 递归式

### 递归二叉查找

```python

def __bitGet(self, number, n):
        return (((number)>>(n)) & 0x01)  

    def findBinNumberLost(self, array):
        '''
        找出所缺失的整数
        '''
        length = len(array)
        A = deepcopy(array)
        B = arange(0, length + 1, dtype=float)
        for i in range(length + 1):
            B[i] = math.inf
        for i in range(length):
            # 禁止使用A[i]
            B[A[i]] = A[i]
        for i in range(length + 1):
            if B[i] == math.inf:
                return i

    def __findNumUsingBinTreeRecursive(self, array, rootIndex, number):
        root = deepcopy(rootIndex)
        if root < 0 or root >= len(array):
            return False
        if array[root] == number:
            return True
        elif array[root] > number:
            return self.__findNumUsingBinTreeRecursive(array, root - 1, number)
        else:
            return self.__findNumUsingBinTreeRecursive(array, root + 1, number)

    def findNumUsingBinTreeRecursive(self, array, number):
        '''
        在排序好的数组中使用递归二叉查找算法找到元素

        Args
        =
        array : a array like, 待查找的数组
        number : a number, 待查找的数字

        Return
        =
        result :-> boolean, 是否找到

        Example
        =
        >>> Chapter4_4().findNumUsingBinTreeRecursive([1,2,3,4,5], 6)
        >>> False

        '''
        middle = (int)(len(array) / 2);
        return self.__findNumUsingBinTreeRecursive(array, middle, number)

```
