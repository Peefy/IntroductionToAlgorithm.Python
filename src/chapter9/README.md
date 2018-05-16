
## 选择问题的分治思路(类似于快速排序)

```python

    def minimum(self, A : list) -> float:
        '''
        求集合中的最小值
        '''
        min = A[0]
        for i in range(1, len(A)):
            if min > A[i]:
                min = A[i]
        return min

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

```

