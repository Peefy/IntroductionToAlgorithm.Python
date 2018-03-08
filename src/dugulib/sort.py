
'''
排序算法集合
'''

def insertsort(array):
    '''
    Summary
    =
    插入排序的升序排列,时间复杂度:O(n^2):
    
    Parameter
    =
    array : a list like
    Return
    =
    sortedarray : 排序好的数组
    >>> import sort
    >>> array = [1, 3, 5, 2, 4, 6]
    >>> sort.insertsort(array)
    >>> [1, 2, 3, 4, 5, 6]
    '''
    A = array
    n = len(A)
    for j in range(1, n):
        ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
        # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
        # 后来摸的牌
        key = A[j]
        # 之前手中的已经排序好的牌的最大索引
        i = j - 1
        # 开始寻找插入的位置并且移动牌
        while(i >= 0 and A[i] > key):
            # 向右移动牌
            A[i + 1] = A[i]
            # 遍历之前的牌
            i = i - 1
        # 后来摸的牌插入相应的位置
        A[i + 1] = key
    # 输出升序排序后的牌
    return A

def selectsort(array):
    '''
    Summary
    =
    选择排序的升序排列,时间复杂度:O(n^2):
    
    Parameter
    =
    array : a list like
    Return
    =
    sortedArray : 排序好的数组
    >>> import sort
    >>> array = [1, 3, 5, 2, 4, 6]
    >>> sort.selectsort(array)
    >>> [1, 2, 3, 4, 5, 6]
    '''
    A = array
    length = len(A)
    for j in range(length):
        minIndex = j
        # 找出A中第j个到最后一个元素中的最小值
        # 仅需要在头n-1个元素上运行
        for i in range(j, length):
            if A[i] <= A[minIndex]:
                minIndex = i
        # 最小元素和最前面的元素交换
        min = A[minIndex]
        A[minIndex] = A[j]
        A[j] = min
    return A

def __bubbleSort(array, start, end):
    A = deepcopy(array)
    p = deepcopy(start)
    q = deepcopy(end)
    if p > q:
        raise Exception('The start index must be less than the end index')
    length = q + 1
    for i in range(p, length):
        for j in range(i + 1, length):
            if A[j] < A[j - 1]:
                # 禁止python的方便写法：A[j], A[j - 1] = A[j - 1], A[j]
                # temp = A[j]
                # A[j] = A[j - 1]
                # A[j - 1] = temp
                A[j], A[j - 1] = A[j - 1], A[j]
    return A

def bubblesort(array):
    '''
    冒泡排序，时间复杂度:o(n^2):

    Args
    ====
    array : 排序前的数组

    Return
    ======
    sortedArray : 使用冒泡排序排好的数组
        
    Example:
    >>> import sort
    >>> A = [6, 5, 4, 3, 2, 1]
    >>> sort.bubbleSort(A)
    >>> [1, 2, 3, 4, 5, 6]
    '''
    return __bubbleSort(array, 0, len(array) - 1)