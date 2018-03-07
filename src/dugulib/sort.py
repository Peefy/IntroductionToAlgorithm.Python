
'''
排序算法集合
'''

def insertSortAscending(self, array = []):
    '''
    Summary
    =
    插入排序的升序排列,时间复杂度O(n^2)
    
    Parameter
    =
    array : a list like
    Return
    =
    sortedArray : 排序好的数组
    >>> array = [1, 3, 5, 2, 4, 6]
    >>> Chapter2().insertSortAscending(array)
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


