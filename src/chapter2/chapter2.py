
## python src/chapter_2/chapter2.py
## python3 src/chapter_2/chapter2.py

import sys
import numpy as nm
from numpy import arange
import matplotlib as mat
import matplotlib.pyplot as plt

class Chapter2:

    def __init__(self, ok = 1, *args, **kwargs):       
        '''
        Summary
        =
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        self

        Example
        =
        >>> chapter2 = Chapter2(ok = 1);
        '''
        self.ok = ok

    def insertSortAscending(self, array = []):
        '''
        Summary
        =
        插入排序的升序排列
        
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

    def insertSortDescending(self, array = []):
        '''
        Summary
        =
        插入排序的降序排列

        Parameter
        =
        array : a list like

        Return
        =
        sortedArray : 排序好的数组
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> Chapter2().insertSortAscending(array)
        >>> [6, 5, 4, 3, 2, 1]
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
            while(i >= 0 and A[i] < key):
                # 向右移动牌
                A[i + 1] = A[i]
                # 遍历之前的牌
                i = i - 1
            # 后来摸的牌插入相应的位置
            A[i + 1] = key
        # 输出降序排序后的牌
        return A

    def arrayContains(self, array = [], v = None):
        '''
        Summary
        =
        * a function
        * *检测一个数组中是否包含一个元素*

        Parameter
        =
        *array* : a list like
        v : a element

        Return
        =
        index:若找到返回找到的索引，没找到返回None

        Example:
        =
        >>> array = [12, 23, 34, 45]
        >>> v = 23
        >>> m = 55
        >>> Chapter2().arrayContains(array, v)
        >>> 1
        >>> Chapter2().arrayContains(array, m)
        >>> None
        '''
        index = None
        length = len(array)
        for i in range(length):
            if v == array[i]:
                index = i
        return index

    def twoNBinNumAdd(self, A = [], B = []):
        '''
        Summary
        =
        两个存放数组A和B中的n位二进制整数相加

        Parameter
        ====
        A : a list like and element of the list must be 0 or 1
        B : a list like and element of the list must be 0 or 1

        Return
        ======
        returnSum : sum of two numbers

        Example:
        =
        >>> A = [1, 1, 0, 0]
        >>> B = [1, 0, 0, 1]
        >>> Chapter2().twoNBinNumAdd(A, B)
        >>> [1, 0, 1, 0, 1]
        '''
        if len(A) != len(B):
            raise Exception('length of A must be equal to length of B')
        length = len(A)
        # 注意：range 函数和 arange 函数都是 左闭右开区间
        '''
        >>> range(0,3) 
        >>> [0, 1, 2]
        '''
        returnSum = arange(length + 1)
        bitC = 0
        for i in range(length):
            index = length - 1 - i
            bitSum = A[index] + B[index] + bitC
            if bitSum >= 2:
                bitSum = 0
                bitC = 1
            else:
                bitC = 0
            returnSum[index + 1] = bitSum
            if index == 0:
                returnSum[0] = bitC
        return returnSum

    def selectSortAscending(self, array = []):
        '''
        Summary
        =
        选择排序的升序排列
        
        Parameter
        =
        array : a list like
        Return
        =
        sortedArray : 排序好的数组
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> Chapter2().selectSortAscending(array)
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

    def note(self, *args, **kwargs):
        '''
        Summary
        =
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        self

        Example
        =
        >>> Chapter2().note()
        '''  
        print('排序算法有很多，包括插入排序，冒泡排序，堆排序，归并排序，选择排序，计数排序，基数排序，桶排序，快速排序')
        print('2.1 插入排序')
        print('插入排序(INSERTION-SORT):输入n个数，输出n个数的升序或者降序排列')
        print('插入排序是一个对少量元素进行排序的有效算法，工作做原理与打牌摸牌整理手中的牌差不多')
        print('以下是Python的插入排序(升序)算法(模拟打牌)')
        print('书中的伪代码数组索引从1开始，python数组索引从0开始')
        A = [4, 4.5, 2, 5, 1.2, 3.5]
        print("待排序的序列：", A)
        print("插入排序后的序列：", self.insertSortAscending(A))
        print('循环不变式主要用来帮助理解插入算法的正确性。证明循环不变式的三个性质')
        print(' 1.初始化：在循环的第一轮迭代开始前，应该是正确的')
        print(' 2.保持：如果在循环的某一次迭代开始之前它是正确的，那么在下一次迭代开始前，它也应该保持正确')
        print(' 3.终止：当循环结束时，不变式给了我们一个有用的性质，有助于表明算法是正确的')
        print('数学归纳法中，要证明某一性质是成立的，必须首先证明其基本情况和一个归纳步骤都是成立的')
        print('插入排序的循环不变式证明：')
        print(' 1.初始化：插入排序第一步首相将数组中第二个元素当做待插入的元素，被插入的元素只有数组中第一元素，显然一个元素是已经排序好的')
        print(' 2.保持：证明每一轮循环都能时循环不变式保持成立,同时证明外层for循环和内层while循环同时满足循环不变式')
        print(' 3.终止：当j大于n时，外层循环结束，新的数组包含了原来数组中的元素，并且是排序好的，算法正确')
        print('布尔运算符and和or都具有短路运算能力')
        print('练习2.1-1：对于序列[31, 41, 59, 26, 41, 58]首先选出序列中第二个元素41向前插入(升序)，接下来选出59向前插入，依次类推')
        print('练习2.1-2：只要把书中插入排序中的伪代码的不等号方向更换即可')
        A = [31, 41, 59, 26, 21, 58]
        print('  排序好的降序序列为：', self.insertSortDescending(A))
        print('练习2.1-3：结果如下:')
        print('  32在序列A中的索引为(索引从0开始)：', self.arrayContains(A, 32))
        print('  21在序列A中的索引为(索引从0开始)：', self.arrayContains(A, 21))  
        print('练习2.1-4：两个n位二进制数相加的算法(适用于FPGA中。参考一位加法器的逻辑表达式或者数学表达式)：')
        print(' 两个n位二进制数的和为：', self.twoNBinNumAdd([1, 1, 0, 1], [1, 0, 0, 0]))
        # range函数和np.arange函数都是左闭右开区间
        print('range(0,4)的值为：', [i for i in range(0,4)])
        print('2.2 算法分析')
        print('算法分析即指对一个算法所需要的资源进行预测')
        print('内存，通信带宽或者计算机硬件等资源是关心的资源, 通常资源指我们希望测度的计算时间')
        print('采用单处理器、随机存取机RAM计算模型')
        print('RAM模型包含了真实计算机中常见的指令：算数指令(加法，减法，除法，取余，向下取整，向上取整指令)，数据移动指令(装入、存储、复制)和控制指令(条件和非条件转移、子程序调用和返回指令)')
        print('RAM模型中的数据类型有整数类型和浮点实数类型')
        print('算法分析所需要的数学工具包括组合数学、概率论、代数')      
        print('需要对"运行时间"和"输入规模"更仔细地加以定义')
        print('插入排序算法的分析')
        print('插入排序INSERTION=SORT过程的时间开销与输入有关，排序1000个数的事件比排序三个数的时间要长')
        print('插入算法即使对给定规模的输入，运行时间也有可能依赖于给定的是该规模下的哪种输入')
        print('插入排序当输入是最好情况(即输入的序列已经按顺序排好)，插入排序所需要的时间随输入规模是线性的O(n)')
        print('插入排序当输入是按照逆序排序的(降序排列输入后输出升序排列),就会出现最坏情况,所需要时间是输入规模的二次函数O(n^2)')
        print('一般考察算法的最坏情况运行时间')
        print('当然对于一些"随机化"算法，其行为即使对于固定的输入，运行时间也是可以变化的')
        print('做进一步的抽象：即运行时间的增长率，只考虑算法运行时间公式中的最高次项，并且忽略最高次项的常数系数,时间复杂度')
        print('练习题2.2-1: n^3/1000 - 100n^2 - 100n + 3 的时间复杂度：O(n^3)')
        print('练习题2.2-2:选择排序如下：')
        A = [21, 11, 9, 66, 51, 48]
        print(' 选择排序排列前的元素：', A)
        print(' 选择排序排列后的元素：', self.selectSortAscending(A))
        print(' 选择排序最好情况o(n^2),最坏情况o(n^2)')
        print(' 因为在第n-1次比较选择的时候已经比较出了最大元素和次大元素并选择，所以这时选择完之后第n个元素已经是最大值，没有必要再比较下去了')
        print('练习题2.2-3:线性查找最好情况是o(1),最坏情况是o(n)，平均情况是(n)')
        print('要使算法具有较好的最佳情况运行时间就一定要对输入进行控制，使之偏向能够使得算法具有最佳运行情况的排列。')

        #python src/chapter_2/chapter2.py
        #python3 src/chapter_2/chapter2.py
        return self

if __name__ == '__main__':
    print('Run main : single chapter two!')
    Chapter2().note()
else:
    print('if you want to run single test, please in your cmd input as follow:\n' + 
        'python src/chapter_2/chapter2.py or \n' + 
        'python3 src/chapter_1/chapter1.py')
    print()

## python src/chapter_2/chapter2.py
## python3 src/chapter_2/chapter2.py

