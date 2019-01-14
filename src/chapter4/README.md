
## 递归式

* 代换法：猜测某个界存在，然后利用数学归纳法证明该猜测的正确性
* 递归树方法 : 将递归式转换成树形结构，最后利用对和式限界的技术来解递归式，n个叶子结点的二叉树高有lgn层
* 主方法：给出递归形式T(n)=aT(n/b) + f(n),其中a>=1,b>=1,f(n)是给定的函数

等比数列求和公式：(1-q)^n / (1-q)

## 主定理

T(n) = aT(n / b) + f(n); a>=1, b>=1

直觉是f(n)和函数n^(logba)进行比较由较大的决定,并且是多项式渐进比较大小

* 对于e>0, 如果有f(n) = O(n^(logba - e)), 则T(n) = Θ(n^(logba))
* 如果有f(n) = Θ(n^(logba)), 则T(n) = n^(logba)lgn
* 对于e>0, 如果有f(n) = Ω(n^(logba + e)), 且对c<1,存在足够大的n使得af(n/b)<=cf(n),则T(n) = Θ(f(n))

> 主定理并没有覆盖所有的情况，并且是多项式渐进比较

如 T(n) = 3T(n/4) + nlgn 满足主定理第3种情况，并且满足规则性条件

但是 T(n) = 2T(n/2) + nlgn 不满足主定理条件，不是多项式大于

## 合并排序

平均时间复杂度O(nlgn), 最坏情况复杂度O(nlgn), 最好情况复杂度O(nlgn), 空间复杂度O(n), 非原地排序，稳定算法

```python

def merge(a, b):
    ret = []
    i = j = 0
    while len(a) >= i + 1 and len(b) >= j + 1:
        if a[i] <= b[j]:
            ret.append(a[i])
            i += 1
        else:
            ret.append(b[j])
            j += 1
    if len(a) > i:
        ret += a[i:]
    if len(b) > j:
        ret += b[j:]
    return ret

def mergesort(arr):
    if len(arr) < 2:
        return arr 
    else: 
        left = mergesort(arr[0 : len(arr) // 2])
        right = mergesort(arr[len(arr) // 2:])
        return merge(left, right)

```

## 分治策略

* 分解：子问题与原问题的形式相同,只是规模更小
* 解决：**递归**地解各子问题,若子问题足够小,则直接求解,求解时间为O(1)
* 合并：将子问题的结果合并成原问题的解

T(n) = Θ(1);   n <= x
T(n) = aT(n/b) + D(n) + C(n); otherwise

子问题的规模也可能不相等

## 最大子数组问题

输入：具有n个数的向量x；
输出：输入向量的任何连续子向量的最大和

### Θ(n)的算法

```python

def arraymaxsub(A):
    maxsofar = 0
    maxendinghere = 0
    for i, val in enumrate(A):
        maxendinghere = max(maxendinghere + val, 0)
        maxsofar = max(maxsofar, maxendinghere)
    return maxsofar

```

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
