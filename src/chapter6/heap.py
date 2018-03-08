'''
:二叉堆:的一系列操作
'''

import math as _math
from numpy import arange as _arange

def left(i):
    '''
    求:二叉堆:一个下标i的:左儿子:的下标
    '''
    return int(2 * i + 1)

def right(i):
    '''
    求:二叉堆:一个下标i的:右儿子:的下标
    '''
    return int(2 * i + 2)

def parent(i):
    '''
    求:二叉堆:一个下标i的:父节点:的下标
    '''
    return (i + 1) // 2 - 1

def heapsize(A):
    '''
    求一个数组形式的:二叉堆:的:堆大小:
    '''
    return len(A) - 1

def maxheapify(A, i):
    '''
    保持堆使某一个结点i成为 :最大堆: (前提条件是其:子树:本身已经为:最大堆:), 时间代价为:O(lgn):

    See Also
    =
    >>> heap.maxheapify_quick

    '''
    l = left(i)
    r = right(i)
    largest = 0
    if  l <= heapsize(A) and A[l] >= A[i]:
        largest = l
    else:
        largest = i
    if r <= heapsize(A) and A[r] >= A[largest]:
        largest = r
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        maxheapify(A, largest)
    return A

def maxheapify_quick(A, i):
    '''
    保持堆使某一个结点i成为最大堆(其子树本身已经为最大堆) :不使用递归算法:
 
    See Also
    =
    >>> heap.maxheapify

    '''
    count = len(A)
    largest = count
    while largest != i:
        l = left(i)
        r = right(i)
        if  l <= heapsize(A) and A[l] >= A[i]:
            largest = l
        else:
            largest = i
        if r <= heapsize(A) and A[r] >= A[largest]:
            largest = r
        if largest != i:
            A[i], A[largest] = A[largest], A[i]
            i, largest = largest, count
    return A

def minheapify(A, i):
    '''
    保持堆使某一个结点i成为:最小堆:(其子树本身已经为:最小堆:)
    '''
    l = left(i)
    r = right(i)
    minest = 0
    if  l <= heapsize(A) and A[l] <= A[i]:
        minest = l
    else:
        minest = i
    if r <= heapsize(A) and A[r] <= A[minest]:
        minest = r
    if minest != i:
        A[i], A[minest] = A[minest], A[i]
        minheapify(A, minest)
    return A

def buildmaxheap(A):
    '''
    对一个数组建立最大堆的过程, 时间代价为:O(n):
    '''
    count = int(len(A) // 2)
    for i in range(count + 1):
        maxheapify(A, count - i)
    return A

def heapsort(A):
    '''
    堆排序算法过程, 时间代价为:O(nlgn):

    Args
    =
    A : 待排序的数组A

    Return
    =
    sortedA : 排序好的数组

    Example
    =
    >>> heap.heapsort([7, 6, 5, 4, 3, 2, 1])
    >>> [1, 2, 3, 4, 5, 6, 7]

    See Also
    =
    >>> heap.buildmaxheap
    >>> heap.maxheapify
    >>> heap.maxheapify_quick
    '''
    heapsize = len(A) - 1

    def __maxheapify(A, i):
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

    buildmaxheap(A)
    length = len(A)   
    for i in range(length - 1):
        j = length - 1 - i
        A[0], A[j] = A[j], A[0]
        heapsize = heapsize - 1
        __maxheapify(A, 0)
    return A
        
def extractmax(A):
    '''
    去掉集合A中具有最大关键字的元素重新构建最大堆,运行时间为:O(lgn):

    Args
    =
    A : 待去掉最大元素的集合A

    Return：
    =
    max : 去掉的最大元素

    '''
    heapsizeA = heapsize(A)
    if heapsizeA < 1:
        raise Exception('heap underflow')
    max = A[0]
    A[0] = A[heapsizeA]
    heapsizeA = heapsizeA - 1
    maxheapify(A, 0)
    return max

def increasekey(A, i, key):
    '''
    将索引为`i`的关键字的值加到`key`，这里`key`的值不能小于索引为`i`原关键字的值并重新构建:最大堆:

    Args
    =
    A : 待操作的集合A
    i : 索引
    key : 提升后的值

    Return
    =
    A : 操作完成的集合A

    Example
    ==
    >>> import heap
    >>> heap.increasekey([4,3,2,1],1,5)
    >>> [5,3,2,1]
    >>> heap.increasekey([4,3,2,1],2,5)
    >>> [5,4,3,1]

    '''
    if key < A[i]:
        raise Exception('new key is smaller than current key')
    A[i] = key
    # 构建最大堆
    while i > 0 and A[parent(i)] < A[i]:
        A[i], A[parent(i)] = A[parent(i)], A[i]
        i = parent(i)
    return A

def maxheapinsert(A, key):
    '''
    向最大堆中插入一个值为`key`的元素，并重新构成:最大堆:

    Args
    =
    A : 待插入元素的数组
    key : 待插入元素的值

    Return
    ==
    A : 插入完成的元素

    '''
    heapsizeA = heapsize(A) + 1
    A.append(-_math.inf)
    increasekey(A, heapsizeA, key)
    return A

def maxheapdelete(A, i):
    '''
    删除一个最大堆索引为`i`的元素：运行代价为:O(nlgn):

    Args
    =
    A : 待操作的数组A
    i : 待删除的索引i

    Return
    =
    A : 删除操作完成后的元素

    Example
    =
    >>> import heap
    >>> heap.maxheapdelete([4,3,2,1],0)
    >>> [3,2,1]

    See Also
    =
    >>> heap.maxheapinsert

    '''
    heapsizeA = heapsize(A) - 1
    count = len(A)
    if i >= count:
        raise Exception('the arg i must not i >= len(A)!')
    A[i] = -_math.inf
    maxheapify(A, i)
    A.pop()
    return A

def buildmaxheap_usesort(A):
    '''
    将数组A构建成为一个:最大堆:
    '''
