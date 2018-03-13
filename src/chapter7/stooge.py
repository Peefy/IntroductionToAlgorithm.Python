
from copy import deepcopy as _deepcopy

def __stoogesort(A, i, j):
    if A[i] > A[j]:
        A[i], A[j] = A[j], A[i]
    if i + 1 >= j:
        return A
    k = (j - i + 1) // 3
    __stoogesort(A, i, j - k)
    __stoogesort(A, i + k, j)
    __stoogesort(A, i, j - k)

def stoogesort(A):
    '''
    Stooge原地排序 时间复杂度为:O(n^2.7):
    '''
    __stoogesort(A, 0, len(A) - 1)

