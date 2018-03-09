

# python src/chapter6/young.py
# python3 src/chapter6/young.py

import math as _math

def array2youngmatrix(A, m, n):
    '''
    将一个数组`A`变成用数组表示的`m`*`n`young矩阵

    Args
    =
    A : 待操作的数组A
    m : 矩阵A的总行数
    n : 矩阵A的总行数

    Return
    =
    A : Young矩阵

    Example
    =
    >>> import young
    >>> young.array2youngmatrix([9, 16, 3, 2, 4, 8, 5, 14, 12], 3, 3)
    >>> [2, 3, 8, 4, 9, 14, 5, 12, 16]
    '''
    count = len(A)
    for i in range(count):
        minyoungify(A, count - 1 - i, m, n)
    return A 

def down(i, m, n):  
    '''
    求一个`m`*`n`矩阵`A`索引`i`元素下方元素的索引，若下方没有元素，返回正无穷
    '''
    if i >= m * n - n :
        return _math.inf
    return i + 1 * n

def right(i, m, n):
    '''
    求一个`m`*`n`矩阵`A`索引`i`元素右方元素的索引，若右方没有元素，返回正无穷
    '''
    index = i + 1
    if index % n == 0:
        return _math.inf
    return i + 1

def minyoungify(A, i, m, n):
    '''
    检查`m`*`n`矩阵`A`处于`i`位置的元素是否小于其右边和下边的元素，如果不小于则交换，
    前提是其右下的元素已经排成了young形式

    Args
    =
    A : 待操作的矩阵
    i : 待操作的元素索引
    m : 矩阵A的总行数
    n : 矩阵A的总行数

    Return
    =
    A : 索引i右下方是Young矩阵的矩阵

    Example
    =
    >>> import young
    >>> young.minyoungify([1, 2, 9, 4, 5, 6, 7, 8], 2, 2, 4)
    >>> [1, 2, 4, 8, 5, 6, 7, 9]
    '''
    count = len(A)
    minest = count
    while minest != i:
        d = down(i, m, n)
        r = right(i, m, n)
        if r < count and A[r] <= A[i]:
            minest = r
        else:
            minest = i
        if d < count and A[d] <= A[minest]:
            minest = d
        if minest != i:
            A[i], A[minest] = A[minest], A[i]
            i, minest = minest, count
    return A

def youngsort(A):
    '''
    使用young矩阵(不利用其他算法)对数组`A`进行排序，时间复杂度:O(n^3):

    Args
    =
    A : 待排序的数组

    Return
    =
    A : 排序好的数组

    Example
    =
    >>> import young
    >>> young.youngsort([9, 8, 7, 6, 5, 4, 3, 2, 1])
    >>> [1, 2, 3, 4, 5, 6, 7, 8, 9]

    '''
    return array2youngmatrix(A, 1, len(A))

# python src/chapter6/young.py
# python3 src/chapter6/young.py

if __name__ == '__main__':
    print(minyoungify([1, 2, 9, 4, 5, 6, 7, 8], 2, 2, 4))
    print(array2youngmatrix([9, 16, 3, 2, 4, 8, 5, 14, 12], 3, 3))
else:
    pass
