'''
二叉堆的一系列操作
'''

def left(i):
    return int(2 * i + 1)

def right(i):
    return int(2 * i + 2)

def parent(i):
    return (i + 1) // 2 - 1

def heapsize(A):
    return len(A) - 1

def maxheapify(A, i):
    '''
    保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
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
    保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
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
    保持堆:使某一个结点i成为最小堆(其子树本身已经为最小堆)
    '''
    l = left(i)
    r = right(i)
    minest = 0
    if  l <=heapsize(A) and A[l] <= A[i]:
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
    对一个数组建立最大堆的过程
    '''
    count = int(len(A) // 2)
    for i in range(count + 1):
        maxheapify(A, count - i)
    return A



