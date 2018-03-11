
'''
排序算法集合
'''

# python src/dugulib/sort.py
# python3 src/dugulib/sort.py

from copy import deepcopy as _deepcopy
import sys as _sys

if __name__ == '__main__':
    _sys.path.append("..")
    from ..chapter6.heap import heapsort
else:
    from ..chapter6.heap import heapsort


class Sort:
    '''
    排序算法集合类
    '''
    def insertsort(self, array):
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

    def selectsort(self, array = []):
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

    def bubblesort(self, array):
        '''
        冒泡排序,时间复杂度:O(n^2):

        Args
        ====
        array : 排序前的数组

        Return
        ======
        sortedArray : 使用冒泡排序排好的数组

        Example:
        >>> import sort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> sort.bubblesort(A)
        >>> [1, 2, 3, 4, 5, 6]

        '''
        nums = _deepcopy(array)
        for i in range(len(nums) - 1):    
            for j in range(len(nums) - i - 1):  
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums

    def __mergeSortOne(self, array, p ,q, r):
        '''
        一步合并两堆牌排序算法过程

        Args
        =
        array : a array like

        Returns:
        =
        sortedArray : 排序好的数组

        Raises:
        =
        None
        '''
        # python中变量名和对象是分离的
        # 此时A是array的一个引用
        A = array
        # 求数组的长度 然后分成两堆([p..q],[q+1..r]) ([0..q],[q+1..n-1])
        n = r + 1

        # 检测输入参数是否合理
        if q < 0 or q > n - 1:
            raise Exception("arg 'q' must not be in (0,len(array) range)")
        # n1 + n2 = n
        # 求两堆牌的长度
        n1 = q - p + 1
        n2 = r - q
        # 构造两堆牌(包含“哨兵牌”)
        L = arange(n1 + 1, dtype=float)
        R = arange(n2 + 1, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
        # 加入无穷大“哨兵牌”, 对不均匀分堆的完美解决
        L[n1] = math.inf
        R[n2] = math.inf
        # 因为合并排序的前提是两堆牌是已经排序好的，所以这里排序一下
        # chapter2 = Chapter2()
        # L = chapter2.selectSortAscending(L)
        # R = chapter2.selectSortAscending(R)
        # 一直比较两堆牌的顶部大小大小放入新的堆中
        i, j = 0, 0
        for k in range(p, n):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
        return A

    def __mergeSort(self, array, start, end):
        '''
        合并排序总过程

        Args:
        =
        array : 待排序数组
        start : 排序起始索引
        end : 排序结束索引

        Return:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().mergeSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        '''
        # python一切皆对象和引用，所以要拷贝...特别是递归调用的时候
        r = deepcopy(end)
        p = deepcopy(start)
        if p < r:
            # 待排序序列劈成两半
            middle = int((r + p) / 2)
            q = deepcopy(middle)
            # 递归调用
            # array =  self.__mergeSort(array, start, middle)
            self.__mergeSort(array, p, q)
            # 递归调用
            # array = self.__mergeSort(array, middle + 1, end)
            self.__mergeSort(array, q + 1, r)
            # 劈成的两半牌合并
            # array = self.__mergeSortOne(array, start ,middle, end)
            self.__mergeSortOne(array, p, q, r)
        return array    

    def mergesort(self, array):
        '''
        归并排序/合并排序：最优排序复杂度:n * O(log2(n)):, 空间复杂度:O(n):

        Args:
        =
        array : 待排序的数组

        Returns:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> import sort
        >>> sort.mergesort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__mergeSort(array, 0, len(array) - 1)


_inst = Sort()
insertsort = _inst.insertsort
selectsort = _inst.selectsort
bubblesort = _inst.bubblesort
mergesort = _inst.mergesort
heapsort = heapsort

if __name__ == '__main__':
    # python src/dugulib/sort.py
    # python3 src/dugulib/sort.py
    print(insertsort([8, 7, 6, 5, 4, 3, 2, 1]))
    print(selectsort([8, 7, 6, 5, 4, 3, 2, 1]))
    print(bubblesort([8, 7, 6, 5, 4, 3, 2, 1]))
    print(mergesort([8, 7, 6, 5, 4, 3, 2, 1]))
    print(heapsort([8, 7, 6, 5, 4, 3, 2, 1]))
else:
    pass

