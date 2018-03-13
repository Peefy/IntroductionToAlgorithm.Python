
from copy import deepcopy as _deepcopy
from random import randint as _randint
class QuickSort:

    def partition(self, A, p, r):
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

    def __quicksort(self, A, p, r):
        left = _deepcopy(p)
        right = _deepcopy(r)
        if left < right:
            middle = _deepcopy(self.partition(A, left, right))
            self.__quicksort(A, left, middle - 1)
            self.__quicksort(A, middle + 1, right)

    def quicksort(self, A):
        '''
        快速排序，时间复杂度:o(n^2):,但是期望的平均时间较好:Θ(nlgn):

        Args
        ====
        A : 排序前的数组:(本地排序):

        Return
        ======
        A : 使用快速排序排好的数组:(本地排序):

        Example
        ==
        >>> import quicksort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> quicksort.quicksort(A)
        >>> [1, 2, 3, 4, 5, 6]
        '''
        self.__quicksort(A, 0, len(A) - 1)
        return A

    def randomized_partition(self, A, p, r):
        i = _randint(p, r)
        A[r], A[i] = A[i], A[r]
        return self.partition(A, p, r)

    def __randomized_quicksort(self, A, p, r):
        left = _deepcopy(p)
        right = _deepcopy(r)
        if left < right:
            middle = _deepcopy(self.randomized_partition(A, left, right))
            self.__randomized_quicksort(A, left, middle - 1)
            self.__randomized_quicksort(A, middle + 1, right)
        return A

    def randomized_quicksort(self, A):
        '''
        使用了随机化技术的快速排序，时间复杂度:o(n^2):,但是期望的平均时间较好:Θ(nlgn):

        Args
        ====
        A : 排序前的数组:(本地排序):

        Return
        ======
        A : 使用快速排序排好的数组:(本地排序):

        Example
        =======
        >>> import quicksort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> quicksort.randomized_quicksort(A)
        >>> [1, 2, 3, 4, 5, 6]
        '''        
        return self.__randomized_quicksort(A, 0, len(A) - 1)

_inst = QuickSort()
partition = _inst.partition
quicksort = _inst.quicksort
randomized_quicksort = _inst.randomized_quicksort