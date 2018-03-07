'''
:二叉堆:的一系列操作
'''

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
    求一个数组形式的二叉堆的堆大小
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
        
    


