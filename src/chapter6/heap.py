
def left(i):
    return int(2 * i + 1)

def right(i):
    return int(2 * i + 2)

def parent(i):
    return (i + 1) // 2 - 1

def heapsize(A):
    return len(A) - 1

def maxheapify(self, A, i):
    '''
    保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
    '''
    l = self.__left(i)
    r = self.__right(i)
    largest = 0
    if  l <= self.__heapsize(A) and A[l] >= A[i]:
        largest = l
    else:
        largest = i
    if r <= self.__heapsize(A) and A[r] >= A[largest]:
        largest = r
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        self.maxheapify(A, largest)
    return A

def maxheapify_quick(self, A, i):
    '''
    保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
    '''
    count = len(A)
    largest = count
    while largest != i:
        l = self.__left(i)
        r = self.__right(i)
        if  l <= self.__heapsize(A) and A[l] >= A[i]:
            largest = l
        else:
            largest = i
        if r <= self.__heapsize(A) and A[r] >= A[largest]:
            largest = r
        if largest != i:
            A[i], A[largest] = A[largest], A[i]
            i, largest = largest, count
    return A

def minheapify(self, A, i):
    '''
    保持堆:使某一个结点i成为最小堆(其子树本身已经为最小堆)
    '''
    l = self.__left(i)
    r = self.__right(i)
    minest = 0
    if  l <= self.__heapsize(A) and A[l] <= A[i]:
        minest = l
    else:
        minest = i
    if r <= self.__heapsize(A) and A[r] <= A[minest]:
        minest = r
    if minest != i:
        A[i], A[minest] = A[minest], A[i]
        self.minheapify(A, minest)
    return A
