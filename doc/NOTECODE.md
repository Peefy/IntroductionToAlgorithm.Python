
```py

## python src/chapter_1/chapter1_1.py
## python3 src/chapter_1/chapter1_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

import numpy as nm
from numpy import arange

import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

import sympy as sym
from sympy import symbols, Symbol
from sympy import solve, poly, exp, log

class Chapter1:
    '''
    CLRS 第一章 算法函数和笔记
    '''
    def __init__(self, ok = 1, *args, **kwargs):
        self.ok = ok

    def __str__(self):
        return 'self.ok' + str(self.ok);

    def note(self, *args, **kwargs):
        '''
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        None

        Example
        =
        >>> print('chapter1 note as follow:')
        '''  
        print('1.1 算法')
        print('算法就是定义良好的计算过程，它取一个或一组值作为输入，并产生出一个或一组值作为输出；简单来讲，算法就是一系列步骤，用来将输入数据转换成输出的结果')
        print('矩阵乘法满足结合律 矩阵连乘 使用动态规划满足减小运算复杂度')
        print('取模方程求解')
        print('给定平面上n个点，找出这些点的凸壳，即包含这些点的最小凸变形')
        print('在Internet中，一个路由节点也需要在网络中寻找最短路径')
        print('可以将CLRS当做菜谱来用，本书作者序')
        print('至今没有人能找出NP完全问题的有效解法，但是也没有人能证明NP完全问题的有效问题不存在')
        print('著名的旅行商人问题就是一个NP完全问题')
        print('一些计算问题：排序问题、矩阵相乘顺序问题、；算法运行速度、运行内存、时间复杂度、空间复杂度、')
        print('练习题1. 1-1：学生考试，对所有学生考试成绩做出排序')
        print('练习题1. 1-2：时间复杂度、空间复杂度、正确性、可读性、健壮性')
        print('练习题1. 1-2：时间复杂度、空间复杂度、正确性、可读性、健壮性')
        print('  时间复杂度:')
        print('    算法的时间复杂度是指执行算法所需要的时间。一般来说，计算机算法是问题规模n 的函数f(n)，算法的时间复杂度也因此记做。 ')
        print('  空间复杂度')
        print('    算法的空间复杂度是指算法需要消耗的内存空间。其计算和表示方法与时间复杂度类似，一般都用复杂度的渐近性来表示。同时间复杂度相比，空间复杂度的分析要简单得多。 ')
        print('  正确性')
        print('    算法的正确性是评价一个算法优劣的最重要的标准。')
        print('  可读性')
        print('    算法的可读性是指一个算法可供人们阅读的容易程度。')
        print('  健壮性')
        print('    健壮性是指一个算法对不合理数据输入的反应能力和处理能力，也成为容错性。')
        print('使用下界函数Omega或者上界函数Theta则分别表示算法运行的最快和最慢时间。')
        print('练习题1. 1-3：数组与链表的优缺点；数组:优点：使用方便 ，查询效率 比链表高，内存为一连续的区域 缺点：大小固定，不适合动态存储，不方便动态添加' +
            '链表：优点：可动态添加删除,大小可变;缺点：只能通过顺次指针访问，查询效率低')   
        print('练习题1. 1-4：最短路径问题：SPFA算法、Dijkstra算法；旅行商人问题(组合优化问题):最近邻点法,插入法，贪心法;' + 
            '相似之处都是求最短距离，不同之处是最短路径问题不经过所有的节点，旅行商人问题要经过所有的节点')
        print('  一些启发式算法：遗传算法、模拟退火法、蚁群算法、禁忌搜索算法、贪婪算法和神经网络等')
        print('练习题1. 1-5：一个正数求根号，牛顿迭代法或者泰勒展开法')
        print('1.2 作为一种技术的算法')
        print('计算时间是一种有限的资源，存储空间也是一种有限的资源，有限的资源需要被有效地使用，那么时间和空间上有效的算法就有助于做到这一点')
        print('效率：插入排序算法的排序时间大约等于c * n ** 2，合并排序算法的排序时间大约等于n * math.log2(n)')
        print('系统的总体性能不仅依赖于选择快速的硬件，还依赖于选择有效的算法')
        print('是否拥有扎实的算法知识和技术基础，是区分真正熟练的程序员与新手的一项重要特征')
        print('练习1.2-1:硬件的设计就要用到算法，任何GUI的设计也要依赖于算法，网络路由对算法也有很大的依赖，编译器，解释器和汇编器这些软件都要用到大量算法。')
        print('  算法是当代计算机中用到的大部分技术的核心。拿网络路由算法举例：算法的目的是找到一条从源路由器到目的路由器的“好”路径（即具有最低费用最短时间的路径)')
        print('  基本的路由算法：LS算法或者Dijkstra算法，链路向量选路算法，距离向量算法')
        # 定义numpy的一个数组，速度快
        interval = 0.2
        n = arange(1, 50, interval)
        # for in 遍历求函数值
        y1_2_2 = [8 * i ** 2 for i in n]
        y2_2_2 = [64 * i * math.log2(i) for i in n]   
        # 利用matplotlib仿matlab画图
        plot(n, y1_2_2, n, y2_2_2)
        show()        
        index = [math.floor(i * interval + 1) + 1 for i in range(1, len(y1_2_2)) if y1_2_2[i] <= y2_2_2[i]]
        # 使用仿Java的string.format()写法
        print('练习1.2-2:当n的范围在{}与{}之间时，插入排序的性能要优于合并排序'.format(index[0], index[-1]))        
        n = arange(1, 15, 0.2)
        y1_2_3 = [100 * i ** 2 for i in n]
        y2_2_3 = [2 ** i for i in n]
        # 可以画图验证两条函数曲线的交点
        figure()
        plot(n, y1_2_3, n, y2_2_3)
        show()
        index = [math.floor(i * interval + 1) + 1 for i in range(1, len(y1_2_3)) if y1_2_3[i] <= y2_2_3[i]]
        print('练习1.2-3:n的最小取值：', index[0])
        n = 1
        t1flag = False;
        t2flag = False;
        t3flag = False;
        t4flag = False;
        t5flag = False;
        t6flag = False;
        t7flag = False;
        t8flag = False;
        while True:
            if t1flag == False:
                t1 = math.log2(n) * 1e-6
            if t2flag == False:
                t2 = math.sqrt(n) * 1e-6
            if t3flag == False:
                t3 = n * 1e-6
            if t4flag == False:
                t4 = n * math.log2(n) * 1e-6
            if t5flag == False:
                t5 = n ** 2 * 1e-6
            if t6flag == False:
                t6 = n ** 3 * 1e-6
            if t7flag == False:
                t7 = 2 ** n * 1e-6
            if t8flag == False:
                t8 = 1 * 1e-6
            for i in range(1, n):
                t8 = t8 * i
            if t8 >= 1 and t8flag == False:
                print('思考题1-1:n!:', n)
                t8flag = True
            if t7 >= 1 and t7flag == False:
                print('思考题1-1:2**n:', n)
                t7flag = True
            if t6 >= 1 and t6flag == False:
                print('思考题1-1:n**3:', n)
                t6flag = True
            if t5 >= 1 and t5flag == False:
                print('思考题1-1:n**2:', n)
                t5flag = True
                break
            n = n + 1
        print('思考题1-1:n*lg(n):太大了循环不过来', )
        print('思考题1-1:lg(n):太大了循环不过来')
        print('思考题1-1:sqrt(n):太大了循环不过来')
        print('思考题1-1:n:1e6')
        return self
        
    def chapter1_1(*args, **kwargs):
        '''
        This chapter1.py main function
        >>> print('hello chapter one!')
        '''  
        print('hello chapter one!')

if __name__ == '__main__':
    print('single chapter one!')
    Chapter1().note()
else:
    pass

## python src/chapter_1/chapter1_1.py
## python3 src/chapter_1/chapter1_1.py
```

```py

## python src/chapter2/chapter2.py
## python3 src/chapter2/chapter2.py
from __future__ import division, absolute_import, print_function
import sys
import numpy as nm
from numpy import arange
import matplotlib as mat
import matplotlib.pyplot as plt

class Chapter2:
    '''
    CLRS 第二章 2.1 2.2 算法函数和笔记
    '''
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

    def __hello():
        pass

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

        #python src/chapter2/chapter2.py
        #python3 src/chapter2/chapter2.py
        return self

if __name__ == '__main__':
    print('Run main : single chapter two!')
    Chapter2().note()
else:
    pass

## python src/chapter2/chapter2.py
## python3 src/chapter2/chapter2.py
```

```py

# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py 
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

if __name__ == '__main__':
    from chapter2 import Chapter2
else:
    from .chapter2 import Chapter2

class Chapter2_3:
    '''
    CLRS 第二章 2.3 算法函数和笔记
    '''

    def insertSortWithIndex(self, array, start ,end):
        '''
        Summary
        ===
        插入排序的升序排列(带排序索引)
        
        Parameter
        ===
        `array` : a list like

        `start` : sort start index

        `end` : sort end index

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> array = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().insert(array, 1, 4)
        >>> [6 ,2, 3, 4, 5, 1]
        ```
        '''
        A = deepcopy(array)
        for j in range(start + 1, end + 1):
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

    def __mergeSortOne2(self, array, p ,q, r):
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
        # 此时A是array的一个深拷贝，改变A不会改变array
        A = deepcopy(array)
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
        L = arange(n1, dtype=float)
        R = arange(n2, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
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
            # 如果L牌堆放完了
            if i == n1:
                # R牌堆剩下的牌数量
                remainCount = n2 - j
                # 把R牌堆剩下的按顺序放到新的牌堆中
                for m in range(remainCount):
                    k += 1
                    A[k] = R[j]
                    j += 1
                break
            # 如果R牌堆放完了
            if i == n2:
                # L牌堆剩下的牌数量
                remainCount = n1 - i
                # 把L牌堆剩下的按顺序放到新的牌堆中
                for m in range(remainCount):
                    k += 1
                    A[k] = L[i]
                    i += 1
                break
        return A

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

    def mergeSort(self, array):
        '''
        归并排序：最优排序复杂度n * O(log2(n)), 空间复杂度O(n)

        Args
        ==
        array : 待排序的数组

        Returns
        ==
        sortedArray : 排序好的数组

        Example
        ==
        >>> Chapter2_3().mergeSort([6, 5, 4, 3, 2, 1])
        
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__mergeSort(array, 0, len(array) - 1)

    def __mergeSortWithSmallArrayInsertSort(self, array, start, end, k):
        p = deepcopy(start)
        r = deepcopy(end)        
        if r - p + 1 > k:
            # 待排序序列劈成两半
            middle = int((r + p) / 2)
            q = deepcopy(middle)
            self.__mergeSortWithSmallArrayInsertSort(array, p, q, k)
            self.__mergeSortWithSmallArrayInsertSort(array, q + 1, r, k)
            self.__mergeSortOne(array, p, q, r)
        return self.insertSortWithIndex(array, p, r)

    def mergeSortWithSmallArrayInsertSort(self, array, k = 4):
        '''
        合并排序 ： 将待排序数组拆分到一定程度(而不是单个元素数组时),子问题足够小时采用插入排序

        Args:
        =
        array : 待排序的数组
        k : 子问题采用插入排序的最小规模

        Returns:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().mergeSortWithSmallArrayInsertSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__mergeSortWithSmallArrayInsertSort(array, 0, len(array) - 1, k)

    def __insertSort(self, array, num):
        key = array[num]
        # 反向查找
        for i in range(num):
            index = num - i - 1
            # 右移           
            if(key <= array[index]):
                array[index + 1] = array[index]
            # 插入
            else:
                array[index + 1] = key
                break
        return array

    def insertSort(self, array, num):
        '''
        递归版本的插入排序

        Args:
        ====
        array : 待排序的数组

        Return: 
        ======
        sortedArray : 排序好的数组

        Example: 
        =
        >>> Chapter2_3().insertSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        '''
        if num > 0:
            # O(1)
            self.insertSort(array, num - 1)  
            # O(n)
            self.__insertSort(array, num)   
        return array

    def factorial(self, n):
        '''
        Factorial of n

        Args:
        =
        n : the factorial number

        Return:
        =
        factorial : the factorial of the number

        Example:
        =
        >>> Chapter2_3().factorial(4)
        >>> 24

        '''
        # Don't forget to terminate the iterations
        if n <= 0:
            return 1
        return n * self.factorial(n - 1)

    def insertSortDichotomy(self, array, index):
        '''
        二分法插入排序

        Args:
        =
        array : 待排序的数组

        Return:
        =
        sortedArray : 排序好的数组

        Example:
        =
        >>> Chapter2_3().insertSortDichotomy([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]

        '''
        A = deepcopy(array)
        n = len(A)
        if index >= n or index < 0 : 
            raise Exception("arg 'index' must be in range [0,len(array))")
        for j in range(1, index + 1):
            ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
            # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
            # 后来摸的牌
            key = A[j]
            # 之前手中的已经排序好的牌的最大索引
            i = j - 1
            low = 0
            high = i
            insertIndex = 0
            # 二分法寻找插入的位置
            while low <= high :
                middle = int((low + high) / 2)
                if key >= A[middle] and key <= A[middle + 1] :
                    insertIndex = middle + 1
                if key > A[middle]:
                    high = middle - 1
                if key < A[middle]:
                    low = middle + 1
            # 移动牌
            while(i >= insertIndex):
                # 向右移动牌
                A[i + 1] = A[i]
                # 遍历之前的牌
                i = i - 1
            # 后来摸的牌插入相应的位置
            A[i + 1] = key
        # 输出升序排序后的牌
        return A
        
    def __sumOfTwoNumbersEqual(self, array ,lastIndex, x):
        n = len(array)
        for i in range(0, lastIndex):
            if abs(array[i] + array[lastIndex] - x) < 10e-5:
                return True
        return False 

    def __internalSumOfTwoNumbersEqual(self, array, index, x):
        isFind = False
        n = deepcopy(index)
        # 如果n<0就结束递归
        if n < 0:
            return
        A = deepcopy(array)
        # 如果存在就结束递归
        result = self.__internalSumOfTwoNumbersEqual(A, n - 1, x)
        if result == True:
            return True
        return self.__sumOfTwoNumbersEqual(A, n, x) 

    def sumOfTwoNumbersEqual(self, array, x):
        '''
        判断出array中是否存在有两个其和等于x的元素

        Args:
        =
        array : 待判断的集合
        x : 待判断的元素

        Return:
        =
        result -> bool : 是否存在

        Example:
        =
        >>> A = [1, 2, 3]
        >>> Chapter2_3().sumOfTwoNumbersEqual(A, 3)
        >>> True
        >>> A = [1, 2, 3, 4]
        >>> Chapter2_3().sumOfTwoNumbersEqual(A, 9)
        >>> False
        '''
        return self.__internalSumOfTwoNumbersEqual(array, len(array) - 1, x)

    def __bubbleSort(self, array, start, end):
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

    def bubbleSort(self, array):
        '''
        冒泡排序，时间复杂度o(n ** 2)

        Args
        ====
        array : 排序前的数组

        Return
        ======
        sortedArray : 使用冒泡排序排好的数组

        Example:
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().bubbleSort(A)
        >>> [1, 2, 3, 4, 5, 6]

        '''
        return self.__bubbleSort(array, 0, len(array) - 1)
    
    def calPolynomial(self, a_array, x):
        '''
        计算多项式

        Args
        ====
        a_array : 多项式系数的数组
        x : 待计算的多项式的代入未知数

        Return
        ======
        y : 计算的多项式的值

        Example
        =
        2x^2 + 2x + 1, x = 2, 2 * 2 ** 2 + 2 * 2 + 1 = 13
        >>> a_array = [2, 2, 1] 
        >>> Chapter2_3().hornerRule(a_array, 2)      
        >>> 13       
        '''
        n = len(a_array)
        total =0
        for i in range(n):
            total = total + a_array[i] * x ** (n - 1 - i)
        return total
  
    def calPolynomialWithHornerRule(self, a_array, x):
        '''
        用霍纳规则计算多项式

        Args
        ====
        a_array : 多项式系数的数组
        x : 待计算的多项式的代入未知数

        Return
        ======
        y : 计算的多项式的值

        Example
        =
        2x^2 + 2x + 1, x = 2, 2 * 2 ** 2 + 2 * 2 + 1 = 13
        >>> a_array = [2, 2, 1] 
        >>> Chapter2_3().hornerRule(a_array, 2)      
        >>> 13       
        '''
        y = 0
        n = len(a_array)
        for i in range(n):
            y = a_array[i] + x * y
        return y

    def __inversion(self, array, end):
        # 进行深拷贝保护变量
        list = deepcopy([])
        n = deepcopy(end)
        A = deepcopy(array)
        if n > 1 :
            newList = self.__inversion(array, n - 1)
            # 相当于C#中的foreach(var x in newList); list.Append(x);
            for i in newList:
                list.append(i)
        lastIndex = n - 1
        for i in range(lastIndex):
            if A[i] > A[lastIndex]:
                list.append((i, lastIndex))
        return list

    def inversion(self, array):
        '''
        递归方式求得数组中的所有逆序对，时间复杂度O(n * lg(n))

        Args
        =
        array : 代求逆序对的数组

        Return
        =
        list : 所有逆序对索引的集合，集合中的每一个元素都为逆序对的元组

        Example
        =
        >>> A = [2, 3, 8, 6, 1]
        >>> Chapter2_3().inversion(A)
        >>> [(0, 4), (1, 4), (2, 3), (2, 4), (3, 4)]      
        '''
        return self.__inversion(array, len(array))

    def inversionListNum(self, array):
        '''
        递归方式求得数组中的所有逆序对，时间复杂度O(n * lg(n))

        Args
        =
        array : 代求逆序对的数组

        Return
        =
        list_num : 所有逆序对索引的集合的长度

        Example
        =
        >>> A = [2, 3, 8, 6, 1]
        >>> Chapter2_3().inversionListNum(A)
        >>> 5     
        '''
        return len(self.inversion(array))

    def note(self):
        '''
        Summary
        =
        Print chapter2.3 note
        Example
        =
        >>> Chapter2_3().note()
        '''
        print('chapter 2.3 note')
        print('算法设计有很多方法')
        print('如插入排序方法使用的是增量方法，在排好子数组A[1..j-1]后，将元素A[j]插入，形成排序好的子数组A[1..j]')
        print('2.3.1 分治法')
        print(' 很多算法在结构上是递归的，所以采用分治策略，将原问题划分成n个规模较小而结构与原问题相似的子问题，递归地解决这些子问题，然后再合并其结果，就得到原问题的解')
        print(' 分治模式在每一层递归上都有三个步骤：分解，解决，合并')
        print(' 分治法的一个例子:合并排序')
        print('  分解：将n个元素分成各含n/2个元素的子序列；')
        print('  解决：用合并排序法对两个子序列递归地排序；')
        print('  合并：合并两个已排序的子序列以得到排序结果')
        print(' 在对子序列排序时，其长度为1时递归结束。单个元素被视为是已经排序好的')
        print(' 合并排序的关键步骤在与合并步骤中的合并两个已排序子序列')
        print(' 引入辅助过程MERGE(A,p,q,r),A是个数组，p,q,r是下标,满足p<=q<r,将数组A拆分成两个子数组A[p,q]和A[q+1,r]')
        print(' 数组A的长度为n = r - p + 1，合并过程的时间代价为O(n)')
        print(' 用扑克牌类比合并排序过程，假设有两堆牌都已经排序好，牌面朝上且最小的牌在最上面，期望结果是这两堆牌合并成一个排序好的输出堆，牌面朝下放在桌上')
        print(' 步骤是从两堆牌的顶部的两张牌取出其中较小的一张放在新的堆中，循环这个步骤一直到两堆牌中的其中一堆空了为止，再将剩下所有的牌放到堆上即可')
        # 合并排序针对的是两堆已经排序好的两堆牌，这样时间复杂度为O(n)
        A = [2.1, 12.2, 45.6, 12, 36.2, 50]
        print('单步合并排序前的待排序数组', A)
        print('单步合并排序后的数组(均匀分堆)', self.__mergeSortOne(A, 0, 2, len(A) - 1))
        B = [23, 45, 67, 12, 24, 35, 42, 54]
        print('单步合并排序前的待排序数组', B)
        print('单步合并排序后的数组(非均匀分堆)', self.__mergeSortOne(B, 0, 2, len(B) - 1))
        print('单步合并排序在两堆牌已经是有序的条件下时间复杂度是O(n),因为不包含双重for循环')
        A = [6, 5, 4, 3, 2, 1]
        # 浅拷贝
        A_copy = copy(A)
        print('总体合并算法应用，排序', A_copy, '结果:', self.mergeSort(A))
        print('2.3.2 分治法分析')
        print(' 当一个算法中含有对其自身的递归调用时，其运行时间可用一个递归方程(递归式)来表示')
        print(' 如果问题的规模足够小，则得到直接解的时间为O(1)')
        print(' 把原问题分解成a个子问题，每一个问题的大小是原问题的1/b，分解该问题和合并解的时间为D(n)和C(n)')
        print(' 递归式：T(n) = aT(n/b) + D(n) + C(n)')
        print('合并排序算法分析')
        print('合并算法在元素是奇数个时仍然可以正确地工作，假定原问题的规模是2的幂次')
        print('当排序规模为1时，需要常量时间，D(n) = O(1)')
        print(' 分解：这一步仅仅是计算出子数组的中间位置，需要常量时间，所以D(n) = O(1)')
        print(' 解决：递归地解两个规模为n/2的子问题，时间为2T(n/2)')
        print(' 合并：在含有n个元素的子数组上，单步分解合并的时间C(n) = O(n)')
        print(' 合并排序算法的递归表达式T(n) = O(1) n = 1;2T(n/2) + O(n)')
        print('最后解得T(n) = O(n * log2(n))')
        print('练习2.3-1：将输入数组A分解为八个数，然后从底层做合并排序，3和41合并为3,41；52和26合并为26,52，再合并为3,26,41,52,后四个数同理，然后做总的合并')
        A = [3, 41, 52, 26, 38, 57, 9, 49]
        print('数组A=[3,41,52,26,38,57,9,49]的合并排序结果为：', self.mergeSort(A))
        print('练习2.3-2：(tip:哨兵牌可以的)')
        A = [3, 41, 52, 16, 38, 57, 79, 99]
        print('数组A=[3, 41, 52, 16, 38, 57, 79, 99]的单步分解合并排序结果为：', self.__mergeSortOne2(A, 0, 2, len(A) - 1))
        print('练习2.3-3：每一步的递推公式 + 上一步的递推公式左右两边*2即可得通式T(n) = nlog2(n)')
        print('练习2.3-4: 递归插入方法')
        A = [1, 2, 3, 5, 6, 4]
        print('数组A=[1, 2, 3, 5, 6, 4]的递归单步插入排序结果为：', self.__insertSort(A, len(A) - 1))
        A = [3, 41, 12, 56, 68, 27, 19, 29]
        print('数组A=[3, 41, 12, 56, 68, 27, 19, 29]的递归插入排序结果为：', self.insertSort(A, len(A) - 1))
        print('递归插入的T(1) = O(1) ; T(n) = T(n-1) + n')
        print('练习2.3-6:二分法插入排序')
        # 插入排序最坏情况
        A = [6, 5, 4, 3, 2, 1]
        print('数组A=[1, 2, 3, 5, 6, 4]的二分法插入排序结果为：', self.insertSortDichotomy(A, len(A) - 1))
        print('阶乘的递归', self.factorial(4))
        print('练习2.3-7(鉴戒并归排序的思路或者阶乘的思路，递归)')
        print('[6,5,4,3,2,1]中找5的结果是：', self.sumOfTwoNumbersEqual(A, 5))
        print('[6,5,4,3,2,1]中找11的结果是：', self.sumOfTwoNumbersEqual(A, 11))
        print('[6,5,4,3,2,1]中找12的结果是：', self.sumOfTwoNumbersEqual(A, 12))
        print('思考题2-1:在并归排序中对小数组采用插入排序')
        print(' 带索引的插入排序如下：')
        print(' [6,5,4,3,2,1]从索引1到4的排序为：', self.insertSortWithIndex(A, 1, 4))
        A = [8, 7, 6, 5, 4, 3, 2, 1]
        print(' [8,7,6,5,4,3,2,1]从小问题采用插入排序的合并排序结果为：', 
            self.mergeSortWithSmallArrayInsertSort(A))
        print(' [8,7,6,5,4,3,2,1]从小问题采用插入排序的合并排序结果为(最小规模为3)：', 
            self.mergeSortWithSmallArrayInsertSort(A, 3))
        print(' 1.最坏的情况下，n/k个子列表(每个子列表的长度为k)可以用插入排序在O(nk)时间内完成排序')
        print(' 2.这些子列表可以在O(nlg(n/k)最坏情况时间内完成合并)')
        print(' 3.修改后的合并排序算法的最坏情况运行时间为O(nk+nlg(n/k)),k的最大渐进值为1/n')
        print(' 4.在实践中，k的值应该按实际情况选取')
        print('思考题2-2:冒泡排序的正确性')
        print(' [8,7,6,5,4,3,2,1]的冒泡排序结果为：', 
            self.bubbleSort(A))
        print('思考题2-3:用于计算多项式的霍纳规则的正确性')      
        print('分别是否采用霍纳规则计算多项式2x^2+2x+1的值分别如下')
        print('不采用霍纳规则(时间复杂度为O(n ** 2))：', self.calPolynomial([2, 2, 1], 2))
        print('采用霍纳规则(时间复杂度为O(n))：', self.calPolynomialWithHornerRule([2, 2, 1], 2))
        print('思考题2-4:逆序对')
        print('逆序对的定义：对于一个数组A，如果数组的索引在i<j的情况下，有A[i] > A[j],则(i,j)就称为数组A中的一个逆序对inversion')
        print('列出数组[2,3,8,6,1]的5个逆序对：', self.inversion([2, 3, 8, 6, 1]))
        print('列出数组[1,2,3,4,5]的逆序对：', self.inversion([1, 2, 3, 4, 5]))
        print('列出数组[5,4,3,2,1]的逆序对：', self.inversion([5, 4, 3, 2, 1]))       
        # python src/chapter2/chapter2_3.py
        # python3 src/chapter2/chapter2_3.py
        return self

if __name__ == '__main__':
    Chapter2_3().note()
else:
    pass

# python src/chapter2/chapter2_3.py
# python3 src/chapter2/chapter2_3.py


```

```py

# python src/chapter3/chapter3_1.py
# python3 src/chapter3/chapter3_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter3_1:
    '''
    CLRS 第三章 3.1 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter3.1 note

        Example
        =
        >>> Chapter3_1().note()
        '''
        print('第3章 函数的增长')
        print(' 对于足够大的输入规模，在精确表示的运行时间中，常数项和低阶项通常由输入规模所决定')
        print(' 当输入规模大到时只与运行时间的增长量级有关时，就是研究算法的渐进效率')
        print(' 从极限的角度看，只关心算法运行时间如何随着输入规模的无限增长而增长')
        print(' 对不是很小的输入规模而言，从渐进意义上说更有效的算法是最佳的选择')
        print('3.1渐进记号')
        print(' 算法最坏情况运行时间T(n),因为T(n)一般仅定义于整数的输入规模上')
        print(' Θ O Ω o ω五种记号 ') 
        print(' Θ记号：在第二章中，知道插入排序的最坏情况下运行时间是T(n)=Θ(n^2)')
        print(' 对一个给定的函数，用Θ(g(n))表示一个函数集合')
        print('  数学定义:对于函数f(n):存在正常数c1,c2和n0，使对所有的n >= n0,有0 ≤ c1 * g(n) ≤ f(n) ≤ c2 * g(n)')
        print('  即存在正常数c1,c2；使当n充分大时，f(n)能被夹在c1 * g(n)和c2 * g(n)中间')
        print(' Θ记号渐进地给出一个函数的上界和下界(渐进确界)，当只有渐进上界时，使用O记号。对一个函数g(n),用O(g(n)表示一个函数集合)')
        print(' O记号是用来表示上界的，当用它作为算法的最坏情况运行时间的上界，就对任意输入有运行时间的上界')
        print(' 例子：插入排序在最坏情况下运行时间的上界O(n^2)也适用于每个输入的运行时间。')
        print(' 但是，插入排序最坏情况运行时间的界Θ(n^2)并不是对每种输入都适用。当输入已经排好序时，插入排序的运行时间为Θ(n)')
        print(' 正如O记号给出一个函数的渐进上界，Ω记号给出函数的渐进下界。给定一个函数g(n),用Ω(g(n))表示一个函数集合')
        print(' Ω记号描述了渐进下界，当它用来对一个算法最佳情况运行时间限界时，也隐含给出了在任意输入下运行时间的界。')
        print(' 例如：插入排序的最佳情况运行时间是Ω(n),隐含着该算法的运行时间是Ω(n)')
        print(' 定理3.1 对任意两个函数f(n)和g(n)，f(n)=Θ(g(n))当且仅当f(n)=O(g(n))和f(n)=Ω(g(n))')
        print(' 插入排序的运行时间介于Ω(n)和O(n^2)之间，因为它处于n的线性函数和二次函数的范围内')
        print(' 插入排序的运行时间不是Ω(n^2),因为存在一个输入(当输入已经排好序时)，使得插入排序的运行时间为Ω(n^2)')
        print(' 当说一个算法的运行时间(无修饰语)是Ω(g(n))时，是指对每一个n值，无论取该规模下什么样的输入，该输入上的运行时间都至少是一个常数乘上g(n)(当n足够大时)')
        print('等式和不等式中的渐进符号')
        print(' 合并排序的最坏情况运行时间表示为递归式：T(n)=2T(n/2)+Θ(n)')
        print(' 一个表达式中的匿名函数的个数与渐进记号出现的次数是一致的。')
        print(' 有时渐进记号出现在等式的左边，例如：2n^2+3n+1=2n^2+Θ(n)=Θ(n^2)')
        print('o记号:O记号所提供的渐进上界可能是也可能不是渐进紧确的。界2n^2=O(n^2)是渐进紧确的，但2n=O(n^2)却不是')
        print(' 使用o记号来表示非渐进紧确的上界，例如2n=o(n^2),但是2n^2≠o(n^2)')
        print('ω记号与Ω记号的关系就好像o记号与O记号的关系一样，用ω记号来表示非渐进紧确的下界')
        print('小写的渐进记号表示不是渐进紧确的界')
        print('例如：n^2/2=ω(n),但n^2/2≠ω(n^2)')
        print('函数间的比较')
        print('假设f(n)和g(n)是渐进正值函数')
        print(' 1.传递性：')
        print('   f(n) = Θ(g(n)) 和 g(n) = Θ(h(n)) 推出f(n) = Θ(h(n))')
        print('   f(n) = O(g(n)) 和 g(n) = O(h(n)) 推出f(n) = O(h(n))')
        print('   f(n) = Ω(g(n)) 和 g(n) = Ω(h(n)) 推出f(n) = Ω(h(n))')
        print('   f(n) = o(g(n)) 和 g(n) = o(h(n)) 推出f(n) = o(h(n))')
        print('   f(n) = ω(g(n)) 和 g(n) = ω(h(n)) 推出f(n) = ω(h(n))')
        print(' 2.自反性：')
        print('   f(n) = Θ(f(n)) 和 f(n) = O(f(n)) 推出f(n) = Ω(f(n))')
        print(' 3.对称性：')
        print('   f(n) = Θ(f(n)) 当且仅当 g(n) = Θ(f(n))')
        print(' 4.转置对称性：')
        print('   f(n) = O(g(n)) 当且仅当 g(n) = Ω(f(n))')
        print('   f(n) = o(g(n)) 当且仅当 g(n) = ω(f(n))')
        print('   可以将两个函数f与g的渐进比较和两个实数a与b的比较作一类比')
        print('    f(n) = O(g(n)) ≈ a ≤ b')
        print('    f(n) = Ω(g(n)) ≈ a ≥ b')
        print('    f(n) = Θ(g(n)) ≈ a = b')
        print('    f(n) = o(g(n)) ≈ a < b')
        print('    f(n) = ω(g(n)) ≈ a > b')
        print(' 5.三分性：')
        print('   对两个实数a和b，下列三种情况恰有一种成立：a < b; a = b; a > b')
        print('   虽然任何两个实数都可以作比较，但并不是所有的函数都是可渐进可比较的')
        print('   亦即，对于两个函数f(n)和g(n),可能f(n) = O(g(n))和f(n) = Ω(g(n))都不成立,',
            '例如函数n和n^(1+sin(n))无法利用渐进记号来比较，因为n^(1+sin(n))中的指数值在0到2之间变化')
        print('练习3.1-1:取c1=0.5；c2=2即可 容易证明')
        print('练习3.1-2:三联不等式三边同时除以n^b不等式不变号')
        print('练习3.1-3:将上界O符号和至少放在一起没有意义，一个是小于等于，一个是大于等于')
        print('练习3.1-4：2^(n+1)=O(2^n)成立；2^(2n)=O(2^n)不成立')
        print('练习3.1-5：容易证明')
        print('练习3.1-6：容易证明')
        print('练习3.1-7：证明同a > b 和a < b的交集是空集')
        print('练习3.1-8：略')
        # python src/chapter3/chapter3_1.py
        # python3 src/chapter3/chapter3_1.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter three!')
    Chapter3_1().note()
else:
    pass

```

```py

# python src/chapter3/chapter3_2.py
# python3 src/chapter3/chapter3_2.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show


class Chapter3_2:
    '''
    CLRS 第三章 3.2 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter3.2 note

        Example
        =
        >>> Chapter3_2().note()
        '''
        print('3.2 标准记号和常用函数')
        print('单调性：一个函数f(n)是单调递增的，若m<=n,则有f(m)<=f(n)，反之单调递减，将小于等于号换成小于号，即变为严格不等式，则函数是严格单调递增的')
        print('下取整(floor)和上取整(ceiling)')
        print('取模运算(modular arithmetic)')
        print('多项式定义及其性质')
        print('指数式定义及其性质')
        print('任何底大于1的指数函数比任何多项式函数增长得更快')
        print('对数定义及其性质')
        print('阶乘定义及其性质')
        print('计算机工作者常常认为对数的底取2最自然，因为很多算法和数据结构都涉及到对问题进行二分')
        print('任意正的多项式函数都比多项对数函数增长得快')
        print('斯特林近似公式：n!=sqrt(2*pi*n)*(n/e)^n*(1+Θ(1/n))')
        print('阶乘函数的一个更紧确的上界和下界：')
        print('n!=o(n^n) n!=ω(2^n) lg(n!)=Θ(nlgn)')
        print('函数迭代的定义和性质')
        print('多重对数函数：用记号lg * n(读作n的log星)来表示多重对数，定义为lg * n=min(i>=0;lg^(i)n<=1)')
        print('多重函数是一种增长很慢的函数')
        print('lg * 2 = 1; lg * 4 = 2; lg * 16 = 3; lg * 65536 = 4; lg * 2^65536 = 5')
        print('宇宙中可以观察到的原子数目估计约有10^80，远远小于2^65536,因此很少会遇到一个使lg * n > 5的一个n输入规模')
        print('斐波那契数列：F0 = 0 F1 = 1 F(i) = F(i-1) + F(i-2),产生的序列为0,1,1,2,3,5,8,13,21,34,55,……')
        print('斐波那契数列和黄金分割率φ以及共轭有关系')
        print('φ=((1+sqrt(5))/2=1.61803 和它的共轭(1-sqrt(5))/2=-0.61803)')
        print('练习题和思考题略')
        # python src/chapter3/chapter3_2.py
        # python3 src/chapter3/chapter3_2.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter three!')
    Chapter3_2().note()
else:
    pass

```

```py

# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_1:
    '''
    CLRS 第四章 4.1 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.1 note

        Example
        =
        >>> Chapter4_1().note()
        '''
        print('第四章 递归式')
        print('当一个算法包含对自身的递归调用时，其运行时间通常可以用递归式来表示，递归式是一种等式或者不等式')
        print('递归式所描述的函数是用在更小的输入下该函数的值来定义的')
        print('本章介绍三种解递归式子的三种方法,找出解的渐进界Θ或O')
        print('1.代换法：先猜测某个界存在，再用数学归纳法猜测解的正确性')
        print('2.递归树方法：将递归式转换为树形结构')
        print('3.主方法：给出递归形式T(n)=aT(n/b)+f(n),a>=1;b>1;f(n)是给定的函数')
        print('4.1 代换法')
        print('代换法解递归式的两个步骤，代换法这一名称源于当归纳假设用较小值时，用所猜测的值代替函数的解',
            '，这种方法很有效，但是只能用于解的形式很容易猜的情形')
        print(' 1.猜测解的形式')
        print(' 2.用数学归纳法找出使解真正有效的常数')
        print('代换法可用来确定一个递归式的上界或下界。')
        print('例子：确定递归式 T(n)=2T([n/2])+n 的一个上界，首先猜测解为T(n)=O(nlgn),然后证明T(n)<cnlgn (上界的定义);c>0')
        print('假设这个界对[n/2]成立，即T([n/2])<=c[n/2]lg([n/2])<=cnlg(n/2)+n=cnlgn-cnlg2+n=cnlgn-cn+n<=cnlgn')
        print('最后一步只要c>=1就成立')
        print('接下来应用数学归纳法就要求对边界条件成立。一般来说，可以通过证明边界条件符合归纳证明的基本情况来说明它的正确性')
        print('对于递归式 T(n)=2T([n/2])+n ，必须证明能够选择足够大的常数c, 使界T(n)<=cnlgn也对边界条件成立')
        print('假设T(1)=1是递归式唯一的边界条件。那么对于n=1时，界T(n)<=cnlgn也就是T(1)<=c1lg1=0,与T(1)=1不符')
        print('因此，归纳证明的基本情况不能满足')
        print('对特殊边界条件证明归纳假设中的这种困难很容易解决。对于递归式 T(n)=2T([n/2])+n ,利用渐进记号，只要求对n>=n0,证明T(n)<=cnlgn,其中n0是常数')
        print('大部分递归式，可以直接扩展边界条件，使递归假设对很小的n也成立')
        print('不幸的是，并不存在通用的方法来猜测递归式的正确解，猜测需要经验甚至是创造性的')
        print('例如递归式 T(n) = 2T([n/2]+17)+n 猜测T(n)=O(nlgn)')
        print('猜测答案的另一种方法是先证明递归式的较松的上下界，因为递归式中有n，而我们可以证明初始上届为T(n)=O(n^2)')
        print('然后逐步降低其上界，提高其下界，直至达到正确的渐进确界T(n)=Θ(nlgn)')
        print('例子：T(n)=T([n/2])+T([n/2])+1 ')
        print('先假设解为T(n)=O(n),即要证明对适当选择的c，有T(n)<=cn')
        print('T(n)<=c[n/2]+c[n/2]+1=cn+1,但是无法证明T(n)<=cn,所以可能会猜测一个更大的界,如T(n)=O(n^2),当然也是一个上界')
        print('当然正确的解是T(n)=O(n)')
        print('避免陷阱：在运用渐进表示时很容易出错，例如T(n)<=2(c[n/2])+n<=cn+n=O(n),因为c是常数，因而错误地证明了T(n)=O(n)')
        print('错误在与没有证明归纳假设的准确形式')
        print('变量代换：有时对一个陌生的递归式作一些简单的代数变换，就会使之变成熟悉的形式，考虑T(n)=2T([sqrt(n)])+lgn,令m=lgn即可')
        print('练习4.1-1：使用代换法，假设不等式成立，即T(n)<=clgn;c>0，当然也有T([n/2])<=clg([n/2])')
        print(' 所以T(n)=T([n/2])+1<=clgn-c+1,所以只要c取的足够大就能使n>=2均使不等式成立，得证')
        print('练习4.1-2：使用代换法，假设不等式成立，即T(n)<=clgn;c>0，当然也有T([n/2])<=clg([n/2])')
        print(' T(n)=2T([n/2])+n<=cnlgn-cnlg2+n=cnlgn-cn+n<=cnlgn;当且仅当c>=1成立')
        print(' 再使用代换法证明T(n)>=clgn;c>0, T(n)=2T([n/2])+n>=cnlgn-cnlg2+n=cnlgn-cn+n>=cnlgn;当且仅当c<1时成立')
        print(' 所以递归的解的确定界为Θ(nlgn)')
        print('练习4.1-3：略')
        print('练习4.1-4：略')
        print('练习4.1-5：略')
        print('练习4.1-6：令m=lgn')
        # python src/chapter4/chapter4_1.py
        # python3 src/chapter4/chapter4_1.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_1().note()
else:
    pass

```

```py

# python src/chapter4/chapter4_2.py
# python3 src/chapter4/chapter4_2.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_2:
    '''
    CLRS 第四章 4.2 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.2 note

        Example
        =
        >>> Chapter4_2().note()
        '''
        print('4.2 递归树方法')
        print('虽然代换法给递归式的解的正确性提供了一种简单的证明方法，但是有的时候很难得到一个好的猜测')
        print('就像分析合并排序递归式那样，画出一个递归树是一种得到好猜测的直接方法。在递归树中，每一个结点都代表递归函数调用集合中的一个子问题的代价')
        print('将树中每一层内的代价相加得到一个每层代价的集合，再将每层的代价相加得到递归是所有层次的总代价')
        print('当用递归式表示分治算法的运行时间时，递归树的方法尤其有用')
        print('递归树最适合用来产生好的猜测，然后用代换法加以验证。')
        print('但使用递归树产生好的猜测时，通常可以容忍小量的不良量，因为稍后就会证明')
        print('如果画递归树时非常的仔细，并且将代价都加了起来，那么就可以直接用递归树作为递归式解的证明')
        print('建立一颗关于递归式 T(n)=3T(n/4)+cn^2 的递归树;c>0,为了方便假设n是4的幂，根部的cn^2项表示递归在顶层时所花的代价')
        print('如果递归树代价准确计算出就可以直接作为解，如果不准确计算出代价就可以为代换法提供一个很好的假设')
        print('不准确计算出递归树代价的一个例子： T(n)=T(n/3)+T(2n/3)+O(n)')
        print('为了简化起见，此处还是省略了下取整函数和上取整函数，使用c来代表O(n)项的常数因子。当将递归树内各层的数值加起来时，可以得到每一层的cn值')
        print('从根部到叶子的最长路径是n->(2/3)n->(2/3)^2n->...->1')
        print('因为当k=log3/2(n)时，(2/3)^kn=1,所以树的深度是log3/2(n)')
        print('如果这颗树是高度为log3/2(n)的完整二叉树，那么就有2^(log3/2(n))=n^(log3/2(2))个叶子')
        print('由于叶子代价是常数，因此所有叶子代价的总和为Θ(n^(log3/2(2))),或者说ω(nlgn)')
        print('然而，这颗递归树并不是完整的二叉树，少于n^(log3/2(2))个叶子，而且从树根往下的过程中，越来越多的内部节点在消失')
        print('因此，并不是所有层次都刚好需要cn代价；越靠近底层，需要的代价越少')
        print('虽然可以计算出准确的总代价，但记住我们只是想要找出一个猜测来使用到代换法中')
        print('容忍这些误差，而来证明上界O(nlgn)的猜测是正确的')
        print('事实上，可以用代换法来证明O(nlgn)是递归式解的上界。下面证明T(n)<=dnlgn,当d是一个合适的正值常数')
        print('T(n)<=T(n/3)+T(2n/3)+cn<=dnlgn;成立条件是d>=c/(lg3-(2/3)),因此没有必要准确地计算递归树中的代价')
        print('练习4.2-1:树的深度为lgn,等比数列求和公式为S=a1(1-q^n)/(1-q),所以树的总代价为Θ(1*n)，对数性质a^(logb(c))=c^(logb(a))')
        print(' 再用代换法验证一下T(n)=3T(n/2)+n<=3cn/2+n显然,T(n)=3T(n/2)+n>=3cn/2+n显然,渐进确界Θ')
        print('练习4.2-2:书中已经证明过了')
        print('练习4.2-3:树的深度为log4(n),证明同4.2-1，渐进确界Θ(n)')
        print('练习4.2-4:略')
        print('练习4.2-5:书中有例子，渐进上界O(nlgn)')
        # python src/chapter4/chapter4_2.py
        # python3 src/chapter4/chapter4_2.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_2().note()
else:
    pass

```

```py

# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show


class Chapter4_3:
    '''
    CLRS 第四章 4.3 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.3 note

        Example
        =
        >>> Chapter4_3().note()
        '''
        print('4.3 主方法')
        print('主方法给出求解形如 T(n) = aT(n/b)+f(n) 的递归式子的解，其中a>=1,b>1,f(n)是一个渐进正的函数')
        print('主方法要求记忆三种情况，这样可以很容易确定许多递归式子的解')
        print('递归式描述了将规模为n的问题划分为a个子问题的算法的运行时间，每个子问题规模为n/b,a和b是正常数')
        print('a个子问题被分别递归地解决，时间各为T(n/b)。划分原问题和合并答案的代价由函数f(n)描述')
        print('如合并排序过程的递归式中有a=2;b=2,f(n)=Θ(n)')
        print('或者将递归式写为T(n) = aT([n/b])+f(n)')
        print('定理4.1(主定理)')
        print(' 1.若对于某常数ε>0,有f(n)=O(n^(logb(a)-ε)),则T(n)=Θ(n^(logb(a)))')
        print(' 2.若f(n)=Θ(n^logb(a)),则T(n)=Θ(n^(logb(a))lgn)')
        print(' 3.若对于某常数ε>0,有f(n)=Ω(n^(logb(a)+ε)),且对常数c<1与所有足够大的n,有af(n/b)<=cf(n),则T(n)=Θ(f(n))')
        print('以上三种情况，都把函数f(n)与函数n^logb(a)进行比较,1中函数n^logb(a)更大，则解为T(n)=Θ(n^(logb(a)))')
        print('而在3情况中，f(n)是较大的函数，则解为Θ(f(n)),在第二种情况中函数同样大，乘以对数因子，则解为T(n)=Θ(n^(logb(a))lgn)')
        print('但是三种情况并没有覆盖所有可能的f(n),如果三种情况都满足，则主方法不能用于解递归式子')
        print('主方法的应用')
        print(' 1.T(n)=9(n/3)+n,对应于主定理中第一种情况T(n)=Θ(n^2)')
        print(' 2.T(n)=(2n/3)+1,对应于主定理中第二种情况T(n)=Θ(lgn)')
        print(' 3.T(n)=3T(n/4)+nlgn,有a=3,b=4,f(n)=nlgn,对应于主定理中第三种情况T(n)=Θ(nlgn)')
        print(' 4.递归式T(n)=2T(n/2)+nlgn对主定理方法不适用 nlgn 渐进大于n，并不是多项式大于，所以落在情况二和情况三之间')
        print('练习4.3-1 a) T(n)=Θ(n^2); b) T(n)=Θ(n^2lgn); c) T(n)=Θ(n^2);')
        print('练习4.3-2 算法A的运行时间解由主定理求得T(n)=Θ(n^2),a最大整数值为16')
        print('练习4.3-3 属于主定理的第二种情况T(n)=Θ(lgn)')
        print('练习4.3-4 不能用主定理给出渐进确界')
        print('练习4.3-5 略')
        # python src/chapter4/chapter4_3.py
        # python3 src/chapter4/chapter4_3.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_3().note()
else:
    pass

```

```py

# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from copy import copy
from copy import deepcopy

from numpy import *
import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_4:
    '''
    CLRS 第四章 4.4 算法函数和笔记
    '''

    def __bitGet(self, number, n):
        return (((number)>>(n)) & 0x01)  

    def findBinNumberLost(self, array):
        '''
        找出所缺失的整数
        '''
        length = len(array)
        A = deepcopy(array)
        B = arange(0, length + 1, dtype=float)
        for i in range(length + 1):
            B[i] = math.inf
        for i in range(length):
            # 禁止使用A[i]
            B[A[i]] = A[i]
        for i in range(length + 1):
            if B[i] == math.inf:
                return i

    def __findNumUsingBinTreeRecursive(self, array, rootIndex, number):
        root = deepcopy(rootIndex)
        if root < 0 or root >= len(array):
            return False
        if array[root] == number:
            return True
        elif array[root] > number:
            return self.__findNumUsingBinTreeRecursive(array, root - 1, number)
        else:
            return self.__findNumUsingBinTreeRecursive(array, root + 1, number)

    def findNumUsingBinTreeRecursive(self, array, number):
        '''
        在排序好的数组中使用递归二叉查找算法找到元素

        Args
        =
        array : a array like, 待查找的数组
        number : a number, 待查找的数字

        Return
        =
        result :-> boolean, 是否找到

        Example
        =
        >>> Chapter4_4().findNumUsingBinTreeRecursive([1,2,3,4,5], 6)
        >>> False

        '''
        middle = (int)(len(array) / 2);
        return self.__findNumUsingBinTreeRecursive(array, middle, number)

    def note(self):
        '''
        Summary
        =
        Print chapter4.4 note

        Example
        =
        >>> Chapter4_4().note()
        '''
        print('4.4 主定理的证明 page45 pdf53 画递归树证明 详细过程略')
        print('思考题4-1')
        print(' a) T(n)=Θ(n)')
        print(' b) T(n)=Θ(lgn)')
        print(' c) T(n)=Θ(n^2)')
        print(' d) T(n)=Θ(n^2)')
        print(' e) T(n)=Θ(n^2)')
        print(' f) T(n)=Θ(n^0.5*lgn)')
        print(' g) T(n)=Θ(lgn)')
        print(' h) T(n)=Θ(nlgn)')
        print('思考题4-2')
        print(' 数组[0,1,2,3,5,6,7]中所缺失的整数为:',
            self.findBinNumberLost([7, 5, 2, 3, 0, 1, 6]))
        print('思考题4-3')
        print('数组[1,2,3,4,5]中是否包含6:', self.findNumUsingBinTreeRecursive([1, 2, 3, 4, 5], 6))
        print('数组[1,2,3,4,5]中是否包含2:', self.findNumUsingBinTreeRecursive([1, 2, 3, 4, 5], 2))
        print('思考题4-4 略')
        print('思考题4-5 略')
        print('思考题4-6 略')
        print('思考题4-7 略')
        
        # python src/chapter4/chapter4_4.py
        # python3 src/chapter4/chapter4_4.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_4().note()
else:
    pass

```

```py

# python src/chapter5/chapter5_1.py
# python3 src/chapter5/chapter5_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from random import randint 

from copy import copy 
from copy import deepcopy 

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter5_1:
    '''
    CLRS 第五章 5.1 算法函数和笔记
    '''

    def myRandom(self, a = 0, b = 1):
        '''
        产生[a,b]之间的随机整数
        '''
        return randint(a, b)

    def myBiasedRandom(self):
        pass

    def note(self):
        '''
        Summary
        =
        Print chapter5.1 note

        Example
        =
        >>> Chapter5_1().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.1 雇佣问题')
        print('假设需要雇佣一个一个新的办公室助理，之前的雇佣都失败了，所以决定找一个雇佣代理，雇佣代理每天推荐一个应聘者')
        print('每找到一个更好地应聘者，就辞掉之前的应聘者')
        print('当然雇佣，面试，开除，给代理中介费都需要一定的\"代价\"')
        print('HIRE-ASSISTANT(n)过程伪代码')
        print(' 1. best <- 0')
        print(' 2. for i <- 1 to n')
        print(' 3.     interview candidate i')
        print(' 4.     if candidate i is better than candidate best')
        print(' 5.          then best <- i')
        print(' 6.              hire candidate i')
        print('关心的重点不是HIRE-ASSISTANT的执行时间，而是面试和雇佣所花的费用')
        print('最坏情况分析')
        print('在最坏情况下，我们雇佣了每个面试的应聘者。当应聘者的资质逐渐递增时，就会出现这种情况，此时我们雇佣了n次，总的费用O(nc)')
        print('事实上既不能得知应聘者的出现次序，也不能控制这个次序。因此，通常我们预期的是一般或平均情况')
        print('概率分析是在问题的分析中应用概率技术，大多数情况下，使用概率分析来分析一个算法的运行时间')
        print('为了进行概率分析，必须使用关于输入分布的知识或对其假设，然后分析算法，计算出一个期望的运行时间')
        print('在所有应聘者的资格之间，存在一个全序关系。因此可以使用从1到n的唯一号码来讲应聘者排列名次')
        print('用rank(i)表示应聘者i的名次，并约定较高的名次对应较有资格的应聘者')
        print('这个有序序列rank(1),rank(2),...,rank(3)是序列1,2,...,n的一个排列')
        print('应聘者以随机的顺序出现，就等于说这个排名列表是数字1到n的n!(n的阶乘)')
        print('或者，也可以称这些排名构成一个均匀的随机排列；亦即在n!中可能的组合中，每一种都以相等的概率出现')
        print('随机算法：为了利用概率分析，需要了解关于输入分布的一些情况。在许多情况下，我们对输入分布知之甚少')
        print('一般的，如果一个算法的输入行为不只是由输入决定，同时也由随机数生成器所产生的数值决定，则称这个算法是随机的')
        print('练习5.1-1:每次HIRE应聘者是有一个顺序的，HIRE的时候同时把次序压如栈中，就得到了排名的总次序')
        random_list = [self.myRandom(), self.myRandom(), self.myRandom(), self.myRandom(), self.myRandom(),]
        print('产生5个[0,1]的随机整数', random_list)
        # python src/chapter5/chapter5_1.py
        # python3 src/chapter5/chapter5_1.py
        return self

_instance = Chapter5_1()
note = _instance.note  

if __name__ == '__main__':
    print('Run main : single chapter five!')
    Chapter5_1().note()
else:
    pass

```

```py

# python src/chapter5/chapter5_2.py
# python3 src/chapter5/chapter5_2.py
from __future__ import division, absolute_import, print_function
import sys as _sys

import math as _math

import random as _random

from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_2:
    '''
    CLRS 第五章 5.2 算法函数和笔记
    '''

    def __inversionListNum(self, array):

        # local function
        def __inversion(array, end):
            # 进行深拷贝保护变量
            list = _deepcopy([])
            n = _deepcopy(end)
            A = _deepcopy(array)
            if n > 1 :
                newList = __inversion(array, n - 1)
                # 相当于C#中的foreach(var x in newList); list.Append(x);
                for i in newList:
                    list.append(i)
            lastIndex = n - 1
            for i in range(lastIndex):
                if A[i] > A[lastIndex]:
                    list.append((i, lastIndex))
            return list

        return len(__inversion(array, len(array)))

    def note(self):
        '''
        Summary
        =
        Print chapter5.2 note

        Example
        =
        >>> Chapter5_2().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.2 指示器随机变量')
        print('指示器随机变量为概率与期望之间的转换提供了一个便利的方法。')
        print('给定一个样本空间S和事件A，那么事件A对应的指示器随机变量I{A}的定义为')
        print('I(A)=1,如果A发生的话；I(A)=0,如果A不发生的话')
        print('引理5-1 给定样本空间S和S中的事件A，令Xa=I{A},则E[Xa]=Pr{A}')
        print('利用指示器随机变量分析雇佣问题')
        print('令X作为一个一个随机变量，其值等于雇佣一个新的办公助理的次数')
        print('特别地，令Xi对应于第i个应聘者被雇佣这个事件的指示器随机变量')
        print('Xi=I{第i位应聘者被雇佣}=1 or 0; 1代表被雇佣，0代表没有被雇佣')
        print('并且X=X1+X2+...+Xn')
        print('应聘者i比从应聘者i-1更有资格的概率是1/i,因此也以1/i的概率被雇佣(注意是1/i不是1/n)')
        print('重点：E[X]=sum(1/i)=lnn+O(1)')
        print('即使面试了n个人，平均看起来，实际上大约只雇佣他们之中的lnn个人')
        print('假设应聘者以随机的次序出现，算法HIRE-ASSISTANT总的雇佣费用为O(clnn)')
        print('练习5.2-1 正好雇佣一次的情况就是雇佣了最佳应聘者rank=n的情况，概率为1/n')
        print('练习5.2-2 正好雇佣两次的情况是第一个人除了rank=n那个人都可以，',
            '第二个人必须是最佳雇佣者 P=1/n*∑1/(n-i) ')
        print('练习5.2-3 掷一次骰子的期望数值是3.5，掷n次骰子的期望数值就是3.5n')
        print('练习5.2-4 (帽子保管问题)还回帽子的情况总共有n!种，每个人能拿到自己帽子的概率是1/n，',
            '期望值也是1/n,那么帽子总数的期望值就是每个人帽子数目期望值相加=1')
        print('练习5.2-5 排列的情况总共有n!中，最少情况是升序排列逆序对个数为0，', 
            '最多情况是降序排列n(n-1)/2')
        print('比如[6,5,4,3,2,1]的逆序对个数为6*5/2=15:', 
            self.__inversionListNum([6, 5, 4, 3, 2, 1]))
        # python src/chapter5/chapter5_2.py
        # python3 src/chapter5/chapter5_2.py
        return self

_instance = Chapter5_2()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_2().note()
else:
    pass

```

```py

# python src/chapter5/chapter5_3.py
# python3 src/chapter5/chapter5_3.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_3:
    '''
    CLRS 第五章 5.3 算法函数和笔记
    '''

    def sortbykey(self, array, keys):
        '''
        根据keys的大小来排序array
        '''
        A = _deepcopy(array)
        length = len(A)
        for j in range(length):
            minIndex = j
            # 找出A中第j个到最后一个元素中的最小值
            # 仅需要在头n-1个元素上运行
            for i in range(j, length):
                if keys[i] <= keys[minIndex]:
                    minIndex = i
            # 最小元素和最前面的元素交换
            min = A[minIndex]
            A[minIndex] = A[j]
            A[j] = min
        return A

    def permute_bysorting(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_bysorting([1, 2, 3, 4])
        '''
        n = len(array)
        P = _deepcopy(array)
        for i in range(n):
            P[i] = _randint(1, n ** 3)
            _time.sleep(0.002)
        return self.sortbykey(array, P)

    def randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().randomize_inplace([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_without_identity(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_without_identity([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n - 1):
            _time.sleep(0.001)
            rand = _randint(i + 1, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array

    def permute_with_all(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_with_all([1, 2, 3, 4])
        '''
        n = len(array)
        for i in range(n):
            _time.sleep(0.001)
            rand = _randint(0, n - 1)
            array[i], array[rand] = array[rand], array[i]
        return array
    
    def permute_by_cyclic(self, array):
        '''
        随机打乱排列一个数组

        Args
        =
        array : 随机排列前的数组

        Return:
        =
        random_array : 随机排列后的数组

        Example 
        =
        >>> Chapter5_3().permute_by_cyclic([1, 2, 3, 4])
        '''
        A = _deepcopy(array)
        n = len(array)
        offset = _randint(0, n - 1)
        A = _deepcopy(array)
        for i in range(n):
            dest = i + offset
            if dest >= n:
                dest = dest - n
            A[dest] = array[i]
        return A

    def note(self):
        '''
        Summary
        =
        Print chapter5.3 note

        Example
        =
        >>> Chapter5_3().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.3 随机算法')
        print('了解输入的分布有助于分析算法平均情况行为，但是许多时候无法得到有关输入分布的信息,因而不可能进行平均情况分析')
        print('但是在这些情况下，可以考虑采用随机算法')
        print('对于诸如雇佣问题之类的问题，假设输入的所有排列都是等可能的往往是有益的，通过概率分析可以设计出随机算法')
        print('不是假设输入的一个分布，而是给定一个分布。特别地，在算法运行之前，先随机地排列应聘者，以加强所有排列都是等可能的这个特性')
        print('概率分析和随机算法的区别')
        print('应聘者是以随机顺序出现的话，则雇佣一个新的办公室助理的期望次数大约是lnn')
        print('注意这个算法是确定性的，对于任何特定的输入，雇佣一个新的办公室助理的次数时钟相同')
        print('这个次数将随输入的变化而改变，而且依赖于各种应聘者的排名')
        print('给定A=(1,2,3,4,5,6)，总是会雇佣6次新的助理，因为后来的每一个都比前一个优秀(rank值大)')
        print('给定A=(6,5,4,3,2,1)，总是只会雇佣1次新的助理')
        print('再来考虑一下先对应应聘者进行排列、再确定最佳应聘者的随机算法')
        print('此时随机发生在算法上而不是输入上')
        print('Random-Hire-Assistant')
        print(' 1. randomly permute the list of candidates')
        print(' 2. best <- 0')
        print(' 3. for i <- 1 to n')
        print(' 4.     interview candidate i')
        print(' 5.     if candidate i is better than candidate best')
        print(' 6.          then best <- i')
        print(' 7.              hire candidate i')
        print(' 凭借算法第一步的改变,建立了一个随机算法，它的性能和假设应聘者以随机次序出现所得到的结果是一致的')
        print('引理5.3 过程Random-Hire-Assistant的期望雇佣费用是O(clnn)')
        print('随机排列数组：许多随机算法通过排列给定的输入数组来使输入随机化')
        print('一个常用的方法是为数组的每个元素A[i]赋一个随机的优先级P[i],', 
            '然后依据优先级对数组A中的元素进行排序，这个过程称为PermuteBySorting')
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4, 5 ,6]))
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4, 5, 6]))
        print('[1, 2, 3, 4, 5, 6]采用PermuteBySorting随机打乱后的一个数组:', 
            self.permute_bysorting([1, 2, 3, 4 ,5, 6]))
        print('引理5.4 假设所有的优先级都是唯一的，过程PermuteBySorting可以产生输入的均匀随机排列')
        print('上述算法产生和原来一样的序列的概率是1/n!,而Hire雇佣问题的输入情况可能有n!中，得证')
        print('产生随机排列的一个更好方法是原地排列给定的数列：RandomizeInPlace;复杂度O(n)')
        print('引理5.5：RandomizeInPlace算法也可以计算出一个均匀随机排列')
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4, 5 ,6]))
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4, 5, 6]))
        print('[1, 2, 3, 4, 5, 6]采用RandomizeInPlace随机打乱后的一个数组:', 
            self.randomize_inplace([1, 2, 3, 4 ,5, 6]))
        print('随机算法通常是解决问题的最简单也是最有效的算法')
        print('练习5.3-1 略')
        print('练习5.3-2 使用随机产生非同一排列的算法[1,2,3,4]的两个随机排列', 
            self.permute_without_identity([1, 2, 3, 4]), 
            self.permute_without_identity([4, 3, 2, 1]))
        print('练习5.3-3 使用随机产生非同一排列的算法[1,2,3,4,5]的两个随机排列', 
            self.permute_with_all([1, 2, 3, 4, 5]), 
            self.permute_with_all([5, 4, 3, 2, 1]))
        print('练习5.3-4 使用随机产生非同一排列的算法[1,2,3,4,5,6]的两个随机排列', 
            self.permute_by_cyclic([1, 2, 3, 4, 5, 6]), 
            self.permute_by_cyclic([6, 5, 4, 3, 2, 1]))
        print(' 上述算法肯定不对啊，物理意义就是把数组元素循环向右平移随机个位置，但是元素的相对位置没变')
        print('练习5.3-5 略')
        print('练习5.3-6 略')
        # python src/chapter5/chapter5_3.py
        # python3 src/chapter5/chapter5_3.py
        return self

_instance = Chapter5_3()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_3().note()
else:
    pass

```

```py

# python src/chapter5/chapter5_4.py
# python3 src/chapter5/chapter5_4.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy

from numpy import arange as _arange


class Chapter5_4:
    '''
    CLRS 第五章 5.4 算法函数和笔记
    '''

    def on_line_maximum(self, k , n):
        score = _arange(n)
        bestscore = -_math.inf
        for i in range(k):
            if score[i] > bestscore:
                bestscore = score[i]
        for i in range(k, n):
            if score[i] > bestscore:
                return i
        return n

    def note(self):
        '''
        Summary
        =
        Print chapter5.4 note

        Example
        =
        >>> Chapter5_4().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.4 概率分析和指示器随机变量的进一步使用')
        print('5.4.1 生日悖论：一个房间里面的人数必须要达到多少。才能有两个人的生日相同的机会达到50%')
        print('出现的悖论就在于这个数目事实上远小于一年中的天数，甚至不足年内天数的一半')
        print('我们用整数1,2,...,k对房间里的人编号，其中k是房间里的总人数。另外不考虑闰年的情况，假设所有年份都有n=365天')
        print('而且假设人的生日均匀分布在一年的n天中，索引生日出现在任意一天的概率为1/n')
        print('两个人i和j的生日正好相同的概率依赖于生日的随机选择是否是独立的')
        print('i和j的生日都落在同一天r上的概率为1/n*1/n=1/n^2,所以两人同一天的概率为1/n')
        print('可以通过考察一个事件的补的方法，来分析k个人中至少有两人相同的概率')
        print('至少有两个人生日相同的概率=1-所有人生日都不相同的概率')
        print('所以问题转化为k个人所有人生日都不相同的概率小于1/2')
        print('k个人生日都不相同的概率为P=1*(n-1)/n*(n-2)/n*...*(n-k+1)/n')
        print('且由于1+x<=exp(x),P<=exp(-1/n)exp(-2/n)...exp(-(k-1)n)=exp(-k(k-1)/2n)<=1/2')
        print('所以当k(k-1)>=2nln2时，结论成立')
        print('所以当一年有n=365天时,至少有23个人在一个房间里面，那么至少有两个人生日相同的概率至少是1/2')
        print('当然如果是在火星上，一年有669个火星日，所以要达到相同效果必须有31个火星人')
        print('利用指示器随机变量，可以给出生日悖论的一个简单而近似的分析。对房间里k个人中的每一对(i,j),1<=i<j<=k')
        print('定义指示器随机变量Xij如果生日相同为1生日不同为0')
        print('根据引理5.1 E[Xij]=Pr{i和j生日相同}=1/n')
        print('令X表示计数至少具有相同生日的两人对数目的随机变量，得X=∑∑Xij,i=1 to n j = i + 1 to n')
        print('E[X]=∑∑E[Xij]=k(k-1)/2n,因此当k(k-1)>=2n时，有相同生日的两人对的对子期望数目至少是1个')
        print('如果房间里面至少有sqrt(2n)+1个人，就可以期望至少有两个人生日相同')
        print('对于n=365，如果k=28,具有相同生日的人的对子期望数值为(28*27)/(2*365)≈1.0356.因此如果至少有28个人')
        print('对于上述两种算法，第一种分析仅利用了概率，给出了为使存在至少一对人生日相同的概率大于1/2所需的人数')
        print('第二种分析使用了指示器随机变量，给出了所期望的相同生日数为1时的人数。虽然两种情况下的准确数目不等，但他们在渐进意义上是相等的，都是Θ(sqrt(n))')
        print('5.4.2 球与盒子')
        print('把相同的球随机投到b个盒子里的过程，其中盒子编号为1,2,...,b。每次投球都是独立的')
        print('球落在任一个盒子中的概率为1/b，因此，投球的过程是一组伯努利实验，每次成功的概率为1/b')
        print('成功是指球落入指定的盒子中。这个模型对分析散列技术特别有用')
        print('落在给定盒子里的球数服从二项分布b(k; n, 1/b)')
        print('如果投n个球，落在给定盒子中的球数的期望值是n/b')
        print('在给定的盒子里面至少有一个球之前，平均至少要投几个球？')
        print('要投的个数服从几何分布，概率为1/b，成功之前的期望个数是1/(1/b)=b')
        print('在期望每个盒子里都有一个球之前，大约要投blnb次，这个问题也称为赠券收集者问题')
        print('意思是一个人如果想要集齐b种不同的赠券中的每一种，大约要有blnb张随机得到的赠券才能成功')
        print('5.4.3 序列')
        print('抛一枚均匀硬币n次,期望看到连续正面的最长序列有多长，答案是Θ(lgn)')
        print('5.4.4 在线雇佣问题')
        print('考虑雇佣问题的一个变形。假设现在我们不希望面试所有的应聘者来找到最好的一个，',
            '也不希望因为不断有更好地申请者出现而不停地雇佣新人解雇旧人')
        print('更愿意雇佣接近最好的应聘者，只雇佣一次')
        print('每次面试后，必须或者立即提供职位给应聘者，活着告诉他们没被录用')
        print('在最小化面试次数和最大化雇佣者的质量两方面如何取得平衡')
        print('建模过程：令score(i)表示给第i个应聘者的分数，并且假设没有两个应聘者的分数相同')
        print('在面试j个应聘者之后，我们知道其中哪一个分数最高，但是不知道在剩余n-j个应聘者中会不会有更高分数的应聘者')
        print('采用这样一个策略：选择一个正整数k<n,面试前k个应聘者然后拒绝他们，再雇佣其后比前面的应聘者有更高分数的第一个应聘者')
        print('如果结果是最好的应聘者在前k个面试的之中，那么我们将雇佣第n个应聘者')
        print('OnLineMaximum(k,n):', self.on_line_maximum(5, 10))
        print('练习5.4-1 一个房间里面必须要有多少人，才能让某人和你生日相同的概率至少为1/2 ? ')
        print(' 必须要有多少人，才能让至少两个人生日为7月4日的概率大于1/2')
        print('练习5.4-2')
        print('练习5.4-3')
        print('练习5.4-4')
        print('练习5.4-5')
        print('练习5.4-6')
        print('练习5.4-7')
        print('思考题5-1')
        print('思考题5-2')
        # python src/chapter5/chapter5_4.py
        # python3 src/chapter5/chapter5_4.py
        return self

_instance = Chapter5_4()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_4().note()
else:
    pass

```

```py

# python src/chapter6/chapter6_1.py
# python3 src/chapter6/chapter6_1.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter6_1:
    '''
    CLRS 第六章 6.1 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.1 note

        Example
        =
        >>> Chapter6_1().note()
        '''
        print('第二部分 排序和顺序统计学')
        print('这一部分将给出几个排序问题的算法')
        print('排序算法是算法学习中最基本的问题')
        print('排序可以证明其非平凡下界的问题')
        print('最佳上界可以与这个非平凡下界面渐进地相等，意味者排序算法是渐进最优的')
        print('在第二章中，插入排序的复杂度虽然为Θ(n^2)，但是其内循环是最为紧密的，对于小规模输入可以实现快速的原地排序')
        print('并归排序的复杂度为Θ(nlgn)，但是其中的合并操作不在原地进行')
        print('第六章介绍堆排序，第七章介绍快速排序')
        print('堆排序用到了堆这个数据结构，还要用它实现优先级队列')
        print('插入排序，合并排序，堆排序，快速排序都是比较排序')
        print('n个输入的比较排序的下界就是Ω(nlgn)，堆排序和合并排序都是渐进最优的比较排序')
        print('为研究比较排序算法性能的极限，第八章分析了决策树模型，通过非比较的方式进行排序,则可以突破Ω(nlgn)的下界')
        print('比如计数排序算法，基数排序算法')
        print('顺序统计学')
        print('在由n个数构成的集合上，第i个顺序统计是集合中第i个小的数')
        print('不必有高深的数学知识，但是需要特殊的数学技巧：快速排序，桶排序，顺序统计量悬法')
        print('第六章 堆排序')
        print('堆排序特点：复杂度Θ(nlgn)，原地(in place)排序，利用某种数据结构来管理算法当中的信息')
        print('堆这个词首先是在堆排序中出现，后来逐渐成为\“废料收集存储区\”')
        print('6.1 堆')
        print('(二叉)堆数据结构是一种数组对象，它被视为一颗完全二叉树，树的每一层都是填满的')
        print('表示堆的数组A是一个具有两个属性的对象，', 
            'length[A]是数组中的元素个数,heap-size[A]是存放在A中的堆的元素个数')
        print('虽然A[0..length(A)-1]中都可以包含有效值')
        print('但A[heap-size[A]]之后的元素都不属于相应的堆')
        print('此处length[A]>=heap-size[A],树的根为A[0],给定了某个结点的下标i，可以很轻松的求出其父节点，左儿子和右儿子的下标')
        print('比如下标i的父节点Parent为[i/2],左儿子Left为[2i],右儿子Right为[2i+1]')
        print('一个最大堆(大根堆)可被看作一个二叉树和一个数组')
        print('二叉堆有两种：最大堆和最小堆，最小堆的最小元素是在根部')
        print('最大堆：A[Parent[i]]>=A[i]')
        print('最小堆：A[Parent[i]]<=A[i]')
        print('在堆排序中，使用最大堆，最小堆通常在构造优先队列时使用')
        print('练习6.1-1：在高度为h的堆中，最少元素为1，最多元素为2^h')
        print('练习6.1-2：含n个元素的堆的高度为[lgn]')
        print('练习6.1-3：在一个最大堆的某颗子树，最大元素在该子树的根上')
        print('练习6.1-4：堆的最后的子树的子节点')
        print('练习6.1-5：一个升序排好的数组是一个最小堆')
        print('练习6.1-6：[23,17,14,6,13,10,1,5,7,12]是一个最大堆')
        print('练习6.1-7：当用数组表示了存储n个元素的堆时，叶子节点的下标[n/2]+1,[n/2]+2,...,n')
        # python src/chapter6/chapter6_1.py
        # python3 src/chapter6/chapter6_1.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_1().note()
else:
    pass

```

```py

# python src/chapter6/chapter6_2.py
# python3 src/chapter6/chapter6_2.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
else:
    from . import heap

class Chapter6_2:
    '''
    CLRS 第六章 6.2 算法函数和笔记
    '''

    def maxheapify(self, A, i):
        '''
        保持堆:使某一个结点i成为最大堆(其子树本身已经为最大堆)
        '''
        l = heap.left(i)
        r = heap.right(i)
        largest = 0
        if  l <= heap.heapsize(A) and A[l] >= A[i]:
            largest = l
        else:
            largest = i
        if r <= heap.heapsize(A) and A[r] >= A[largest]:
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
            l = heap.left(i)
            r = heap.right(i)
            if  l <= heap.heapsize(A) and A[l] >= A[i]:
                largest = l
            else:
                largest = i
            if r <= heap.heapsize(A) and A[r] >= A[largest]:
                largest = r
            if largest != i:
                A[i], A[largest] = A[largest], A[i]
                i, largest = largest, count
        return A

    def minheapify(self, A, i):
        '''
        保持堆:使某一个结点i成为最小堆(其子树本身已经为最小堆)
        '''
        l = heap.left(i)
        r = heap.right(i)
        minest = 0
        if  l <= heap.heapsize(A) and A[l] <= A[i]:
            minest = l
        else:
            minest = i
        if r <= heap.heapsize(A) and A[r] <= A[minest]:
            minest = r
        if minest != i:
            A[i], A[minest] = A[minest], A[i]
            self.minheapify(A, minest)
        return A

    def note(self):
        '''
        Summary
        =
        Print chapter6.2 note

        Example
        =
        >>> Chapter6_2().note()
        '''
        print('6.2 保持堆的性质')
        print('Max-heapify是对最大堆操作重要的子程序')
        print('其输入是一个数组A和一个下标i')
        print('假定以Left(i)和Right(i)为根的两颗二叉树都是最大堆，但是这是A[i]可能小于其子女，这样就违反了最大堆的性质')
        print('Max-heapify的过程就是使A[i]下降，使以i为根的子树成为最大堆')
        print('在算法Max-heapify的每一步里，从元素A[i], A[Left[i]], A[Right[i]]找出最大的的下标索引并存在largest中')
        print('如果A[i]已经是最大的，则以i为根的子树已经是最大堆')
        print('以该结点为根的子树又有可能违反最大堆性质，因而又要对该子树递归调用Max-heapify')
        print('Max-heapify的运行时间T(n)<=T(2n/3)+Θ(1)')
        print('根据主定理，该递归式的解为T(n)=O(lgn)')
        print('或者说，Max-heapify作用于一个高度为h的结点所需要的运行时间为O(h)')
        A = [25, 33, 15, 26, 22, 14, 16]
        self.maxheapify(A, 0)
        print('MaxHeapify的一个举例[25,33,15,26,22,14,16]，树的高度为3:', A)
        A = [27, 17, 3, 16, 13, 10, 1, 5, 7, 12, 4,8, 9, 0]
        print('练习6.2-1：在题中索引为3的元素(pyhton中索引为2)为3，写出二叉堆后要使其成为最大堆，',
            '则3和10互换后再和8互换',
            self.maxheapify(A, 2))
        print('练习6.2-2:最小堆的一个例子[3,2,4,11,12,13,14]',
            self.minheapify([3, 2, 4, 11, 12, 13, 14], 0))
        print(' 感觉最大堆保持和最小堆没有区别啊，运行时间一致，只是不等号方向不同')
        print('练习6.2-3，没效果吧，比如[7,6,5,4,3,2,1]:',self.maxheapify([7,6,5,4,3,2,1],0))
        print('练习6.2-4,没效果吧')
        A = [25, 33, 15, 26, 22, 14, 16]
        print('练习6.2-5:[25, 33, 15, 26, 22, 14, 16],', self.maxheapify_quick(A, 0))
        print('练习6.2-6 因为二叉堆树的高度就是lgn，所以最差情况就是遍历了整个二叉堆，运行时间最差为Ω(lgn)')
        # python src/chapter6/chapter6_2.py
        # python3 src/chapter6/chapter6_2.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_2().note()
else:
    pass

```

```py

# python src/chapter6/chapter6_3.py
# python3 src/chapter6/chapter6_3.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
else:
    from . import heap

class Chapter6_3:
    '''
    CLRS 第六章 6.3 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.3 note

        Example
        =
        >>> Chapter6_3().note()
        '''
        print('6.3 建堆')
        print('可以自底向上地使用Max-heapify将一个数组A变成最大堆[0..len(A)-1]')
        print('子数组A[(n/2)+1..n]中的元素都是树中的叶子，因此每个都可以看做是只含一个元素的堆')
        print('BuildMaxHeap的运行时间的界为O(n)')
        print('一个n元素堆的高度为[lgn],并且在任意高度上，至多有[n/2^(h+1)]个结点')
        A = [5,3,17,10,84,19,6,22]
        print('练习6.3-1 BuildMaxHeap作用于数组', _deepcopy(A), 
            "的过程为：", heap.buildmaxheap(A))
        print('练习6.3-2 因为heap.maxheapify使某节点变成最大堆过程的前提是其所有的子树已经是最大堆,所以要先从子树开始，也就是循环下标从大到小')
        print('练习6.3-3 定理：在任一包含n个元素的堆中，至多有[n/2^(h+1)]个高度为h的结点')
        # python src/chapter6/chapter6_3.py
        # python3 src/chapter6/chapter6_3.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_3().note()
else:
    pass

```

```py

# python src/chapter6/chapter6_4.py
# python3 src/chapter6/chapter6_4.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
else:
    from . import heap

class Chapter6_4:
    '''
    CLRS 第六章 6.4 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_4().note()
        '''
        print('6.4 堆排序算法')
        print('设n = len(A)-1,堆排序算法先用BuildMaxHeap将输入数组 A[0..n]构造成一个最大堆')
        print('因为数组中的最大元素在根A[0]，则可以通过把它与A[n]互换来达到最终正确的位置')
        print('现在如果从堆中去掉结点n(通过减小heapsize[A]),可以很容易地将A[1..n-1]建成最大堆，',
            '原来根的子女仍然是最大堆,而新的元素可能违背了最大堆的性质，这时调用MaxHeapify(A, 0)就可以保持这一个性质')
        print('堆排序算法不断重复这个过程，堆的大小由n-1一直降到2')    
        print('堆排序算法的一个举例[7, 6, 5, 4, 3, 2, 1]', heap.heapsort([1, 2, 3, 4, 5, 6, 7]))
        print('HeapSort过程的时间代价O(nlgn)')
        print('调用heap.buildmaxheap的时间为O(n),n-1次heap.maxheapify中每一次的时间代价为O(lgn)')
        A = [5 ,13, 2, 25, 7, 17, 20, 8, 4]
        print('练习6.4-1 数组', _deepcopy(A), '的heapsort过程结果为：', heap.heapsort(A))
        print('练习6.4-2 证明循环不变式的过程略')
        print('练习6.4-3 按递增排序的数组A已经是一个最大堆，buildmaxheap的时间较少，但是交换元素花费时间较多')
        print(' 若A的元素按降序排列，则buildmaxheap的花费时间较多，元素交换时间差不多')
        print('练习6.4-4 略')
        print('练习6.4-5 略')
        # python src/chapter6/chapter6_4.py
        # python3 src/chapter6/chapter6_4.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_4().note()
else:
    pass

```

```py

# python src/chapter6/chapter6_5.py
# python3 src/chapter6/chapter6_5.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
    import young
else:
    from . import heap
    from . import young

class Chapter6_5:
    '''
    CLRS 第六章 6.5 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_5().note()
        '''
        print('6.5 优先级队列')
        print('虽然堆排序算法是一个很漂亮的算法，快速排序(第7章将要介绍)的一个好的实现往往优于堆排序')
        print('堆数据结构还是有着很大的用处，一个很常见的应用：作为高效的优先级队列')
        print('和堆一样，队列也有两种，最大优先级队列和最小优先级队列')
        print('优先级队列是一种用来维护由一组元素构成的集合S的数据结构，这一组元素中的每一个都有一个关键字key')
        print('一个最大优先级队列支持一下操作')
        print('INSERT(S, x):把元素x插入集合S，这一操作可写为S<-S∪{x}')
        print('MAXIMUM(S)：返回S中具有最大关键字')
        print('EXTRACT-MAX(S)：去掉并返回S中的具有最大关键字的元素')
        print('INCREASE-KEY(S，x, k):将元素x的关键字的值增加到k，这里k值不能小于x的原关键字的值')
        print('最大优先级队列的一个应用是在一台分时计算机上进行作业调度。')
        print('当一个作业做完或被中断时，用EXTRACT-MAX操作从所有等待的作业中，选择出具有最高优先级的作业')
        print('在任何时候，一个新作业都可以用INSERT加入到队列中去')
        print('当用堆来实现优先级队列时，需要在堆中的每个元素里存储对应的应用对象的柄handle，',
            '对象柄的准确表示到底怎样(一个指针或者一个整形数)还取决于具体的应用')
        A = [14, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-1 数组A=', _deepcopy(A), '执行HEAP-EXTRACT-MAX操作的过程为：', heap.extractmax(A), A)
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print(' 数组A=', _deepcopy(A), '执行HEAP-INCREASE-KEY操作的过程为：', heap.increasekey(A, 2, 16))
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-2 数组A=', _deepcopy(A), '执行MAX-HEAP-INSERT(A, 10)操作的过程为：', heap.maxheapinsert(A, 10))
        print('练习6.5-3 基本把最大堆算法的不等号方向改以以下就可以')
        print('练习6.5-4 因为插入的元素并不知道其大小和插入前原始最大堆元素的大小比较情况，所以将插入的数据放到二叉堆的最底部的叶子上而且是最小值(负无穷)')
        print('练习6.5-5 略')
        print('练习6.5-6 先进先出队列和栈')
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-7 数组A=', _deepcopy(A), '执行HEAP-DELETE(A, 2)操作的过程为：', heap.maxheapdelete(A, 2))
        print('练习6.5-8 略(不会)')
        A = [1, 2, 3, 4, 5, 6, 7]
        print('思考题6-1 用插入的方法建堆')
        print(' 数组A=', _deepcopy(A), '使用插入方法构建最大堆:', heap.buildmaxheap_usesort(A))
        print('思考题6-2 对d叉堆的分析')
        print(' 含有n个元素的d叉堆的高度为logd(n),或者为lgn/lgd')
        print('思考题6-3 Young氏矩阵')
        print(' Young氏矩阵的元素从左到右，从上到下都是排序好的')
        print(' 将Young氏矩阵斜过来看成一个交叉堆即可')
        A = [9, 16, 3, 2, 4, 8, 5, 14, 12]
        print(' 矩阵A', _deepcopy(A), '变换成一个young矩阵为：', young.array2youngmatrix(A, 3, 3))
        A = [9, 16, 3, 2, 4, 8, 5, 14, 12]
        print(' 矩阵A', _deepcopy(A), '使用young矩阵排序为：', young.youngsort(A))
        # python src/chapter6/chapter6_5.py
        # python3 src/chapter6/chapter6_5.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_5().note()
else:
    pass

```

```py
'''
:二叉堆:的一系列操作
'''

# python src/chapter6/heap.py
# python3 src/chapter6/heap.py
from __future__ import division, absolute_import, print_function
import math as _math
from numpy import arange as _arange

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
    求一个数组形式的:二叉堆:的:堆大小:
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
        
def extractmax(A):
    '''
    去掉集合A中具有最大关键字的元素重新构建最大堆,运行时间为:O(lgn):

    Args
    =
    A : 待去掉最大元素的集合A

    Return：
    =
    max : 去掉的最大元素

    '''
    heapsizeA = heapsize(A)
    if heapsizeA < 1:
        raise Exception('heap underflow')
    max = A[0]
    A[0] = A[heapsizeA]
    heapsizeA = heapsizeA - 1
    maxheapify(A, 0)
    return max

def increasekey(A, i, key):
    '''
    将索引为`i`的关键字的值加到`key`，这里`key`的值不能小于索引为`i`原关键字的值并重新构建:最大堆:

    Args
    =
    A : 待操作的集合A
    i : 索引
    key : 提升后的值

    Return
    =
    A : 操作完成的集合A

    Example
    ==
    >>> import heap
    >>> heap.increasekey([4,3,2,1],1,5)
    >>> [5,3,2,1]
    >>> heap.increasekey([4,3,2,1],2,5)
    >>> [5,4,3,1]

    '''
    if key < A[i]:
        raise Exception('new key is smaller than current key')
    A[i] = key
    # 构建最大堆
    while i > 0 and A[parent(i)] < A[i]:
        A[i], A[parent(i)] = A[parent(i)], A[i]
        i = parent(i)
    return A

def maxheapinsert(A, key):
    '''
    向最大堆中插入一个值为`key`的元素，并重新构成:最大堆:

    Args
    =
    A : 待插入元素的数组
    key : 待插入元素的值

    Return
    ==
    A : 插入完成的元素

    '''
    heapsizeA = heapsize(A) + 1
    A.append(-_math.inf)
    increasekey(A, heapsizeA, key)
    return A

def maxheapdelete(A, i):
    '''
    删除一个最大堆索引为`i`的元素：运行代价为:O(nlgn):

    Args
    =
    A : 待操作的数组A
    i : 待删除的索引i

    Return
    =
    A : 删除操作完成后的元素

    Example
    =
    >>> import heap
    >>> heap.maxheapdelete([4,3,2,1],0)
    >>> [3,2,1]

    See Also
    =
    >>> heap.maxheapinsert

    '''
    heapsizeA = heapsize(A) - 1
    count = len(A)
    if i >= count:
        raise Exception('the arg i must not i >= len(A)!')
    A[i] = -_math.inf
    maxheapify(A, i)
    A.pop()
    return A

def buildmaxheap_usesort(A):
    '''
    将数组A构建成为一个:最大堆:,使用:插入:的方法
    '''
    heapsizeA = 1
    length = len(A)
    B = [A[0]]
    for i in range(1, length):
        maxheapinsert(B, A[i])
    return B

# python src/chapter6/heap.py
# python3 src/chapter6/heap.py

```

```py


# python src/chapter6/young.py
# python3 src/chapter6/young.py
from __future__ import division, absolute_import, print_function
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

```

```py
# python src/chapter7/chapter7note.py
# python3 src/chapter7/chapter7note.py
'''
Class Chapter7_1

Class Chapter7_2

Class Chapter7_3

Class Chapter7_4
'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import quicksort
    import stooge
else:
    from . import quicksort
    from . import stooge

class Chapter7_1:
    def note(self):
        '''
        Summary
        ====
        Print chapter7.1 note

        Example
        ====
        >>> Chapter7_1().note()
        '''
        print('chapter7.1 note as follow')
        print('第7章 快速排序')
        print('快速排序是一种排序算法，对包含n个数的输入数组进行排序，最坏情况的运行时间为Θ(n^2)')
        print('虽然这个最坏情况运行时间比较差，但是快速排序通常是用于排序最佳的实用选择，这是因为其平均性能相当好')
        print('快速排序期望的运行时间为Θ(nlgn),且Θ(nlgn)记号中隐含的常数因子很小')
        print('快速排序能够进行就地排序，在虚存坏境中也能很好地工作')
        print('7.1 快速排序的描述')
        print('像合并排序一样，快速排序也是基于分治模式的')
        print(' 1.分解:数组A[p..r]被划分成两个(可能为空的)子数组A[p..q-1]和A[q+1..r]')
        print('  使得A[p..q-1]中的每个元素都小于等于A(q),而且，小于等于A[q+1..r]')
        print('  下标q也在这个划分过程中进行计算')
        print(' 2.解决:通过递归调用快速排序，对子数组A[p..q-1]和A[q+1..r]排序')
        print(' 3.合并:因为这两个子数组是就地排序的(不开辟新的数组),将他们合并不需要任何操作，整个数组A[p..r]已经排好序')
        print('子数组快速排序伪代码')
        print('QUICKSORT(A,p,r)')
        print(' 1. if q < r')
        print(' 2.   q <- PARTITION(A,p,r)')
        print(' 3.       QUICKSORT(A,p,q-1)')
        print(' 3.       QUICKSORT(A,q+1,r)')
        print('排序完整的数组A，调用QUICKSORT(A,0,len(A))即可')
        print('快速排序算法的关键是PARTITION过程，它对子数组A[q..r]进行就地重排')
        print('PARTITION(A,p,r)')
        print(' 1. x <- A[r]')
        print(' 2. i <- p-1')
        print(' 3. for j <- p to r-1')
        print(' 4.  if A[j] <= x')
        print(' 5.      i <- i+1')
        print(' 6.      exchange A[i] <-> A[j]')
        print(' 7. exchange A[i+1] <-> A[r]')
        print(' 8. return i + 1')
        A = [8, 9, 6, 7, 4, 5, 2, 3, 1]
        print('数组A', _deepcopy(A), '的快速排序过程为:', 
            quicksort.quicksort(A))
        A = [13, 19, 9, 5, 12, 8, 7, 4, 11, 2, 6, 21]
        print('练习7.1-1 数组A', _deepcopy(A), 
            '的一步partition过程得到middle索引为：', 
            quicksort.partition(A, 0, len(A) - 1))
        A = [11, 11, 11, 11, 11]
        print('练习7.1-2 数组A', _deepcopy(A), 
            '的一步partition过程得到middle索引为：', 
            quicksort.partition(A, 0, len(A) - 1))
        print('练习7.1-3 就一个长度为n的for循环，且一定会执行，所以时间复杂度为Θ(n)，然后用确界的夹逼定义证明')
        print('练习7.1-4 不等号方向改变即可')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_2:
    def note(self):
        '''
        Summary
        ====
        Print chapter7.2 note

        Example
        ====
        >>> Chapter7_2().note()
        '''
        print('chapter7.2 note as follow')
        print('7.2 快速排序的性能')
        print('快速排序的运行时间与划分是否对称有关，而后者由与选择了哪个元素来进行划分有关')
        print('如果划分是对称的，那么快速排序算法从渐进上与合并算法一样快，否则就和插入排序一样慢')
        print('快速情况的最坏情况划分行为发生在划分过程产生的两个区域分别包含n-1个元素和1个0元素的时候')
        print('假设每次划分都出现了这种不对称划分，划分的时间代价为Θ(n),故算法的运行时间可以递归地写为')
        print('T(n)=T(n-1)+T(0)+Θ(n),递归式的解为T(n)=Θ(n^2)')
        print('快速排序的最坏情况并不比插入排序的最坏情况更好')
        print('另外，当一个已经排序好时，快速排序运行时间Θ(n^2)，插入排序运行时间Θ(n)')
        print('快速排序最佳情况是是其中一个字问题的大小为[n/2],另一个问题的大小为[n/2]-1')
        print('在这种情况下，快速排序的运行时间要快的多，T(n)<=T(n/2)+Θ(n)')
        print('根据主定理，以上递归式的解为O(nlgn)')
        print('平衡的划分')
        print('快速排序的平均运行时间与其最佳运行时间很接近')
        print('练习7.2-1 T(1)=T(0)+Θ(n),T(2)=T(1)+Θ(n),T(n)=T(n-1)+Θ(n)')
        print(' 从第一个式子加到第n个式子，T(n)=Θ(n^2)')
        print('练习7.2-2 数组A中的每个元素都相同时，也属于元素已经排序好的情况，',
            '所以调用PARTITION子程序每次都会得到最差的分配，所以最坏情况运行时间为T(n)=Θ(n^2)')
        print('练习7.2-3 根据书中的描写降序排序好的元素，会导致每次分配都得到最差的情况，又递归式和主定理得T(n)=Θ(n^2)')
        print('练习7.2-4 对已经排序好的支票对于快速排序来说属于最差情况输入，运行时间为O(n^2)')
        print(' 而对于插入排序来说却是最优输入，运行时间为O(n)')
        print('练习7.2-5 [滑稽]但是平均的运行时间仍然为O(nlgn)')
        print('练习7.2-6 略')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_3:
    '''
    chapter7.3 content : note, function, etc..

    See Also
    ========
    Chapter7_1 Chapter7_2 Chapter7_4
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter7.3 note

        Example
        ====
        >>> Chapter7_3().note()
        '''
        print('chapter7.3 note as follow')
        print('7.3 快速排序的随机化版本')
        print('在探讨快速排序的平均性态过程中，假定输入数据的所有排列都是等可能的')
        print('但在工程中，这个假设就不会总是成立')
        print('虽然第五章介绍过一些随机算法，但是如果采用一种不同的，称为随机取样的随机化技术的话，可以使分析更加简单')
        print('在这种方法中，不是时钟采用A[r]作为主元，而是从子数组A[p..r]中随机选择一个元素')
        print('然后将这个随机元素与A[r]交换作为主元')
        print('因为主元元素是随机选择的，在期望的平均情况下，对输入数组的划分比较对称')
        A = [8, 7, 6, 5, 4, 3, 2, 1]    
        print('数组[8, 7, 6, 5, 4, 3, 2, 1]的随机化快速排序：', 
            quicksort.randomized_quicksort(A))
        print('练习7.3-1:大部分时候输入的待排序序列我们是不知道的，而对于快速排序来讲，一个平均的输入才能反映其算法性能，最坏情况出现的概率比较小')
        print('练习7.3-2:最佳情况调用Θ(n)次，最坏情况调用Θ(n^2)次')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_4:
    '''
    chapter7.4 content : note, function, etc..

    See Also
    ========
    Chapter7_1 Chapter7_2 Chapter7_3
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter7.4 note

        Example
        ====
        ``Chapter7_4().note()``
        '''
        print('chapter7.4 note as follow')
        print('7.4 快速排序分析')
        print('7.4.1 最坏情况分析')
        print('如果快速排序中每一层递归上所做的都是最坏情况划分，则运行时间为Θ(n^2)')
        print('7.4.2 期望的运行时间')
        print('RANDOMZIED-QUICKSORT的平均情况运行时间为O(nlgn)')
        print('运行时间和比较')
        print('quicksort的运行时间是由花在过程PARTITION上的时间所决定的。')
        print('每当PARTITION过程被调用时，就要选出一个主元元素，后续对QUICKSORT和PARTITION的各次递归调用中，都不会包含该元素')
        print('于是，在快速排序算法的整个执行过程中，最多只可能调用PARTITION过程n次，调用一次PARTITION的时间为O(1)在加上一段时间')
        print('引理7.1 设当QUICKSORT在一个包含n个元素的数组上运行时，PARTITION在第四行所做的比较次数为X,那么QUICKSORT的运行时间为O(n+X)')
        print('练习7.4-1 递归式子T(n)=max(T(q)+T(n-q-1)+Θ(n))中，T(n)=Ω(n^2)')
        print('练习7.4-2 快速排序的最佳情况运行时间为Ω(nlgn)')
        print('练习7.4-3 略')
        print('练习7.4-4 RANDOMIZED-QUICKSORT算法期望的运行时间为Ω(nlgn)')
        print('练习7.4-5 对插入排序来说，当其输入已经是几乎排好序的，运行时间是很快的')
        print(' 当在一个长度小于k的子数组上调用快速排序时，让它不做任何排序就返回。', 
            '当顶层的快速排序调用返回后，对整个数组运行插入排序来完成排序过程。', 
            '这一排序算法的期望运行时间为O(nk+nlg(n/k))')
        print('练习7.4-6 PARTITION过程做这样的修改，从数组A中随机地选出三个元素，并围绕这三个数的中数(即这三个元素的中间值)进行划分', 
            '求出以a的函数形式表示的、最坏情况中a:(1-a)划分的近似概率')
        A = [13, 19, 9, 5, 12, 8, 7, 4, 11, 2, 6, 21]
        print('思考题7-1：数组A', _deepcopy(A), '的HOARE-PARTITION算法过程为:', 
            quicksort.hoare_partition(A, 0, len(A) - 1))
        print('数组A', _deepcopy(A), '的HOARE-QUICKSORT的过程为：', quicksort.hoare_quicksort(A))
        print('思考题7-2:对快速排序算法的另一种分析')
        print(' 着重关注每一次QUICKSORT递归调用的期望运行时间，而不是执行的比较次数')
        print(' a) 给定一个大小为n的数组，任何特定元素被选为主元的概率为1/n')
        print('思考题7-3 Stooge排序')
        A = [8, 7, 56, 43, 21]
        print('数组A', _deepcopy(A), '的Stooge排序结果为:', stooge.stoogesort(A), A)
        print('思考题7-4 快速排序的堆栈深度')
        print(' 7.1中的快速排序算法包含有两个对其自身的递归调用,但是第二个递归不是必须的')
        A = [8, 7, 56, 43, 21]
        print('数组A', _deepcopy(A), '的尾递归快速排序结果为:', quicksort.morequicksort(A))
        print('思考题7-5 \"三数取中\"划分 也就是主元素RANDOMIZED-QUICKSORT的RANDOMIZED-PARTITION过程')
        print(' 三数取中方法仅仅影响其运行时间Ω(nlgn)中的常数因子')
        print('思考题7-6 对区间的模糊排序:算法的目标是对这些区间进行模糊排序')
        print('模糊排序算法的期望运行时间为Θ(nlgn),但当所有区间都重叠时，期望的运行时间为Θ(n)')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

chapter7_1 = Chapter7_1()
chapter7_2 = Chapter7_2()
chapter7_3 = Chapter7_3()
chapter7_4 = Chapter7_4()

def printchapter7note():
    '''
    print chapter7 note.
    '''
    print('Run main : single chapter seven!')  
    chapter7_1.note()
    chapter7_2.note()
    chapter7_3.note()
    chapter7_4.note()

# python src/chapter7/chapter7note.py
# python3 src/chapter7/chapter7note.py
if __name__ == '__main__':  
    printchapter7note()
else:
    pass

```

```py

from __future__ import division, absolute_import, print_function

from copy import deepcopy as _deepcopy
from random import randint as _randint
class QuickSort:
    '''
    快速排序相关算程序集合类
    '''
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
        import quicksort
        A = [6, 5, 4, 3, 2, 1]
        quicksort.randomized_quicksort(A)
        [1, 2, 3, 4, 5, 6]
        '''        
        return self.__randomized_quicksort(A, 0, len(A) - 1)

    def hoare_partition(self, A, p, r):
        x = A[p]
        i = p 
        j = r 
        while True:
            while A[j] > x:
                j = j - 1
            while A[i] < x:
                i = i + 1
            if i < j:
                A[i], A[j] = A[j], A[i]
            else:
                return j

    def __hoare_quicksort(self, A, p, r):
        left = _deepcopy(p)
        right = _deepcopy(r)
        if left < right:
            middle = _deepcopy(self.hoare_partition(A, left, right))
            self.__hoare_quicksort(A, left, middle - 1)
            self.__hoare_quicksort(A, middle + 1, right)
        return A

    def hoare_quicksort(self, A):
        '''
        使用了HoarePatition技术的快速排序，时间复杂度:o(n^2):,但是期望的平均时间较好:Θ(nlgn):

        Args
        ====
        A : 排序前的数组:(本地排序):

        Return
        ======
        A : 使用快速排序排好的数组:(本地排序):

        Example
        =======
        ```python
        import quicksort
        >>> A = [6, 5, 4, 3, 2, 1]        
        >>> quicksort.randomized_quicksort(A)       
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''        
        return self.__hoare_quicksort(A, 0, len(A) - 1)

    def __morequicksort(self, A, p, r):
        left = _deepcopy(p)
        right = _deepcopy(r)
        while left < right:
            middle = _deepcopy(self.partition(A, left, right))
            self.__morequicksort(A, left, middle - 1)
            left = middle + 1
        return A
            
    def morequicksort(self, A):
        '''
        使用了尾递归技术的快速排序，最差情况时间复杂度:o(n^2):,
        但是期望的平均时间较好:Θ(nlgn):

        Args
        ====
        A : 排序前的数组:(本地排序):

        Return
        ======
        A : 使用快速排序排好的数组:(本地排序):

        Example
        =======
        ```python
        import quicksort
        A = [6, 5, 4, 3, 2, 1]
        quicksort.randomized_quicksort(A)
        [1, 2, 3, 4, 5, 6]
        ```
        '''       
        return self.__morequicksort(A, 0, len(A) - 1) 

_inst = QuickSort()
partition = _inst.partition
quicksort = _inst.quicksort
randomized_quicksort = _inst.randomized_quicksort
hoare_partition = _inst.hoare_partition
hoare_quicksort = _inst.hoare_quicksort
morequicksort = _inst.morequicksort
```

```py

from __future__ import division, absolute_import, print_function
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


```

```py

# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
'''
Class Chapter8_1

Class Chapter8_2

Class Chapter8_3

Class Chapter8_4
'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter8_1:
    '''
    chpater8.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter8.1 note

        Example
        ====
        ```python
        Chapter8_1().note()
        ```
        '''
        print('chapter8.1 note as follow')
        print('第8章 线性时间排序')
        print('合并排序和堆排序在最坏情况下能达到O(nlgn),快速排序在平均情况下达到此上界面')
        print('第8章之前的排序算法都是比较排序')
        print('8.1节中将证明对含有n个元素的一个输入序列，', 
            '任何比较排序在最坏情况下都要用Ω(nlgn)次比较来进行排序')
        print('由此可知，合并排序和堆排序是最优的')
        print('本章还介绍三种线性时间排序,计数排序，基数排序，桶排序')
        print('8.1 排序算法运行时间的下界')
        print('决策树模型')
        print('比较排序可以被抽象地看作决策树，一颗决策树是一个满二叉树')
        print('在决策树中，每个节点都标有i：j，其中1<=i,j<=n,n是输入序列中元素的个数，控制结构，数据移动等都被忽略')
        print('如排序算法的决策树的执行对应于遍历从树的根到叶子节点的路径')
        print('要使排序算法能正确的工作，其必要条件是n个元素n！种排列中的每一种都要作为一个叶子出现')
        print('对于根结点来说，每一个叶子都可以是某条路径可以达到的')
        print('比较排序算法最坏情况下界，就是从根部到最底部叶子走过的最长路径，也就是树的高度nlgn')
        print('定理8.1 任意一个比较排序在最坏情况下，都需要做Ω(nlgn)次的比较')
        print('堆排序和合并排序都是渐进最优的比较排序算法,运行时间上界O(nlgn)')
        print('练习8.1-1 最小深度可能是n-1，对于已经n个排序好的元素比较n-1次即可，如三个元素比较两次')
        print('练习8.1-2 斯特林近似公式是求n！的一个近似公式')
        print('练习8.1-3 对于长度为n的n!种输入，至少一半而言，不存在线性运行时间的比较排序算法')
        print('练习8.1-4 现有n个元素需要排序，它包含n/k个子序列，每一个包含n个元素')
        print(' 每个子序列的所有元素都小于后续序列的所有元素，所以对n/k个子序列排序，就可以得到整个输入长度的排序')
        print(' 这个排序问题中所需的问题都需要有一个下界Θ(nlgk)')
        print('计数排序的基本思想是对每一个输入元素x，确定出小于x的元素个数，有了这一信息，', 
            '就可以把x直接放到最终输出数组中的位置上。例如有17个元素小于x，则x位于第18个位置上（元素互补相同）')
        print('在计数排序的代码中，假定输入是个数组A[1..n],length[A]=n')
        print('另外还需要两个数组，存放排序结果的B[1..n],以及提供临时存储区的C[0..k]')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_2:
    '''
    chpater8.2 note and function
    '''

    def countingsort2(self, A):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort2([0,1,1,3,4,6,5,3,5])
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        return self.countingsort(A, max(A) + 1)

    def countingsort(self, A, k):
        '''
        针对数组`A`计数排序，无需比较，非原地排序，当`k=O(n)`时，算法时间复杂度为`Θ(n)`,
        3个n for 循环
        需要预先知道数组元素都不大于`k`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B
    
    def note(self):
        '''
        Summary
        ====
        Print chapter8.2 note

        Example
        ====
        ```python
        Chapter8_2().note()
        ```
        '''
        print('chapter8.2 note as follow')
        print('8.2 计数排序')
        print('计数排序假设n个输入元素的每一个都是介于0到k之间的整数，', 
            '此处k为某个整数，k=O(n),计数排序的时间为O(n)')
        A = [5, 5, 4, 2, 1, 0, 3, 2, 1]
        print('数组A:', _deepcopy(A), '的计数排序：', self.countingsort(A, 6))
        print('计数排序虽然时间复杂度低并且算法稳定，但是空间复杂度高，并且需要先验知识`所有元素都不大于k`')
        print('计数排序的稳定性应用非常重要，而且经常作为基数排序的子程序，对于计数排序的正确性证明很重要')
        A = [6, 0, 2, 0, 1, 3, 4, 6, 1, 3, 2]
        print('练习8.2-1 数组A:', _deepcopy(A), '的计数排序：', self.countingsort(A, 7))
        A = [6, 0, 2, 0, 1, 3, 4, 6, 1, 3, 2]
        print(' 数组A:', _deepcopy(A), '另一种计数排序：', self.countingsort2(A))
        print('练习8.2-2 计数算法是稳定的')
        print('练习8.2-3 修改后算法不稳定，最好先放大数再放小数')
        print('练习8.2-4 略 不会')

        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_3:
    '''
    chpater8.3 note and function
    '''
    def getarraystr_subarray(self, A ,k):
        '''
        取一个数组中每个元素第k位构成的子数组

        Args
        ===
        `A` : 待取子数组的数组

        `k` : 第1位是最低位，第d位是最高位

        Return
        ===
        `subarray` : 取好的子数组

        Example 
        ===
        ```python
        Chapter8_3().getarraystr_subarray(['ABC', 'DEF', 'OPQ'], 1)
        ['C', 'F', 'Q']
        ```
        '''
        B = []
        length = len(A)
        for i in range(length):
            B.append(int(str(A[i])[-k]))
        return B
    
    def countingsort(self, A, k):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_2().countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        k = 27
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B

    def radixsort(self, A, d):
        '''
        基数排序 平均时间复杂度为`Θ(nlgn)`

        Args
        ===
        `A` : 待排序的数组

        `d` : 数组A中每个元素都有d位数字/长度,其中第1位是最低位，第d位是最高位

        Return
        ===
        `sortedarray` : 排序好的数组 

        Example
        ===
        ```python
        >>> Chapter8_3().radixsort([54,43,32,21,11], 2)
        >>> [11, 21, 32, 43, 54]
        ```
        '''
        length = len(A)
        B = []
        for i in range(d):
            B.append(self.getarraystr_subarray(A, i + 1))
        for k in range(d):
            B[k] = self.countingsort(B[k], max(B[k]) + 1)
        C = _arange(length)
        for j in range(length):
            for i in range(d):            
                C[j] += B[i][j] * 10 ** i
            C[j] = C[j] - j
        return C 

    def note(self):
        '''
        Summary
        ====
        Print chapter8.3 note

        Example
        ====
        ```python
        Chapter8_3().note()
        ```
        '''
        print('chapter8.3 note as follow')
        print('8.3 基数排序')
        print('基数排序是用在老式穿卡机上的算法')
        print('关于这个算法就是按位排序要稳定')
        print('引理8.3 给定n个d位数，每一个数位有k个可能取值，', 
            '基数排序算法能够以Θ(d(n+k))的时间正确地对这些数排序')
        print('引理8.4 给定n个b位数，和任何正整数r<=b，', 
            'RADIX-SORT能以Θ((b/r)(n+2^r))的时间内正确地排序')
        print('基数排序的时间复杂度表达式中常数项比较大，',
            '若取b=O(lgn),r=lgn,则基数排序的时间复杂度为Θ(n)')
        A = ['ABC', 'DEF', 'OPQ']
        # print('数组A', _deepcopy(A), '的一个取子数组的样例：', self.getarraystr_subarray(A, 1))
        words = [54,43,32,21,11]
        print('练习8.3-1 数组words', _deepcopy(words), '的基数排序为:', self.radixsort(words, 2))
        print('练习8.3-2 排序算法的稳定性：假定在待排序的记录序列中，存在多个具有相同的关键字的记录，',
            '若经过排序，这些记录的相对次序保持不变，即在原序列中，r_i=r_j，且r_i在r_j之前，而在排序后的序列中，', 
            'r_i仍在r_j之前，则称这种排序算法是稳定的，否则成为不稳定的')
        print(' 常见排序算法的稳定性：堆排序，快速排序，希尔排序，直接选择排序不是稳定的排序算法，',
            '而基数排序，冒泡排序，直接插入排序，折半插入排序，合并排序时稳定的排序算法')
        print('练习8.3-3 证明过程中，计数排序要是稳定的才行，不然的话拆分子数组过程在复原后',
            '会导致相应的位数匹配不上，导致结果不正确')
        print('练习8.3-4 使用变形的计数排序，开辟n^2个空间')
        print('练习8.3-5 不会....看管d堆卡片吧')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_4:
    '''
    chpater8.4 note and function
    '''
    def insertsort(self, array):
        '''
        Summary
        ===
        插入排序的升序排列,时间复杂度`O(n^2)`
    
        Parameter
        ===
        `array` : a list like
        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> sort.insertsort(array)
        >>> [1, 2, 3, 4, 5, 6]
        ```
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

    def bucketsort(self, A):
        '''
        桶排序,期望时间复杂度`Θ(n)`(满足输入分布条件`[0,1)`的情况下)
        需要`链表list`额外的数据结构和存储空间

        Args
        ===
        `A` : 待排序的数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_4().bucketsort([0.5, 0.4, 0.3, 0.2, 0.1])
        >>> [0.1, 0.2, 0.3, 0.4, 0.5]
        ```
        '''
        n = len(A)
        B = []
        for i in range(n):
            B.insert(int(n * A[i]), A[i])
        return self.insertsort(B)
        
    def __find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''
        assert len(kettles1) == len(kettles2)
        n = len(kettles1)
        for i in range(n):
            for j in range(n):
                if kettles1[i] == kettles2[j]:
                    yield (i, j)

    def find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''      
        return list(self.__find_matching_kettle(kettles1, kettles2))

    def note(self):
        '''
        Summary
        ====
        Print chapter8.4 note

        Example
        ====
        ```python
        Chapter8_4().note()
        ```
        '''
        print('chapter8.4 note as follow')
        print('当桶排序的输入符合均匀分布时，即可以以线性时间运行，与计数排序类似，桶排序也对输入作了某种假设')
        print('具体来说，计数排序假设输入是由一个小范围内的整数构成，而桶排序则假设输入由一个随机过程产生，该过程将元素均匀地分布在区间[0,1)上')
        print('桶排序的思想就是把区间[0,1)划分成n个相同大小的子区间，或称桶。然后将n个输入分布到各个桶中去')
        print('因为输入数均匀分布在[0,1)上，所以一般不会有很多数落在一个桶中的情况')
        print('为了得到结果，先对桶中的数进行排序，然后按次序把桶中的元素列出来即可。')
        print('在桶排序的算法当中')
        print('桶排序算法中，假设输入的是一个含n个元素的数组A，且每个元素满足[0,1)')
        print('还需要一个辅助数组B[0..n-1]来存放链表(桶)，并假设可以用某种机制来维护这些表')
        print('桶排序的期望时间复杂度为Θ(n),证明过程略')
        A = [0.79, 0.13, 0.16, 0.64, 0.39, 0.20, 0.89, 0.53, 0.71, 0.42]
        print('练习8.4-1 数组A', _deepcopy(A), '的桶排序过程为:', self.bucketsort(A))
        print('练习8.4-2 略')
        print('练习8.4-3 E(X^2)=DX+E^2(X)=9/16+1 E^2(X)=E(X)E(X)=1')
        print('练习8.4-4 所有点到圆心的距离都服从均匀分布，所以采用桶排序')
        print('练习8.4-5 略')
        print('思考题8-1 给定n个不同的输入元素，对于任何确定或随机的比较排序算法，其期望运行时间都有下界Ω(nlgn)')
        print('思考题8-2 以线性时间原地置换排序:假设有一个由n个数据记录组成的数组要排序，且每个记录的关键字的值0或1')
        print(' 算法的运行时间为O(n),算法是稳定的,算法是原地排序的')
        print('思考题8-3 排序不同长度的数据项，字符串所有字符串字符的ascii码都不大于z，所以用基数排序，O(n)')
        print('思考题8-4 水壶：假设给定了n个红色的水壶和n个蓝色的水壶，他们的形状尺寸都不相同，所有红色水壶中所盛水的量都不同。')
        print(' 所有红色水壶中所盛水的量都不一样，蓝色水壶也是一样的；此外，对于每一个红色的水壶，都有一个对应的蓝色水壶，两者所盛的水量是一样的，反之亦然')
        print(' 任务是将匹配的红色水壶和蓝色水壶找出来，假设用1,2,3,4,5代表不同的水量')
        print('[1,2,3,4,5]和[5,4,3,2,1]的匹配为：', 
            self.find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        print(' 只会双重循环比较算法，对于下界为O(nlgn),考虑二叉树和递归算法吧,随机化算法也不会')
        print('思考题8-5 k排序，1排序就是完全排序')
        print(' 一个n元素的数组是k排序的，当且仅当对所有元素，当前A[i]<=A[i+k]')
        print('思考题8-6 合并已排序列表的下界')
        print(' 合并两个已知已排序列表这样的问题是经常出现的。它是合并排序的一个子过程')
        print(' 决策树说明比较次数有一个下界2n-o(n),还有一个更紧确的2n-1界')
        print('Han将排序算法的界改善至O(nlglgnlglglgn),尽管这些算法在理论上有重要的突破，', 
            '但都相当复杂，在目前来看，不太可能与现有的，正在实践中使用的排序算法竞争')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

chapter8_1 = Chapter8_1()
chapter8_2 = Chapter8_2()
chapter8_3 = Chapter8_3()
chapter8_4 = Chapter8_4()

def printchapter8note():
    '''
    print chapter8 note.
    '''
    print('Run main : single chapter eight!')  
    chapter8_1.note()
    chapter8_2.note()
    chapter8_3.note()
    chapter8_4.note()

# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
if __name__ == '__main__':  
    printchapter8note()
else:
    pass

```

```py

# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
'''
Class Chapter9_1

Class Chapter9_2

Class Chapter9_3

'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter9_1:
    '''
    chpater9.1 note and function
    '''

    def minimum(self, A : list) -> float:
        '''
        求集合中的最小值
        '''
        min = A[0]
        for i in range(1, len(A)):
            if min > A[i]:
                min = A[i]
        return min

    def note(self):
        '''
        Summary
        ====
        Print chapter9.1 note

        Example
        ====
        ```python
        Chapter9_1().note()
        ```
        '''
        print('chapter9.1 note as follow')
        print('第9章 中位数和顺序统计学')
        print('在一个由n个元素组成的集合中，第i个顺序统计量是该集合中第i小的元素。')
        print('非形式地说：一个中位数是它所在集合的\"中点元素\"')
        print('当n为奇数时，中位数是唯一的，出现在i=(n+1)/2处；')
        print('当n为偶数时，存在两个中位数，分别出现在i=n/2和i=n/2+1处')
        print('不考虑n的奇偶性，中位数总是出现在i=[(n+1)/2]处(下中位数)和i=[(n+1)/2]处(上中位数)')
        print('简单起见，书中的中位数总是指下中位数')
        print('本章讨论从一个由n个不同数值构成的集合中选择其第i个顺序统计量的问题，假设集合中的数互异')
        print('如下形式化地定义选择问题：')
        print(' 输入：一个包含n个(不同的)数的集合A和一个数i,1<=i<=n')
        print(' 输出：元素x属于A，它恰好大于A中其他i-1个数')
        print('选择问题可以在O(nlgn)时间内解决，因为可以用堆排序或合并排序对输入数据进行排序')
        print(' 然后在输出数组中标出第i个元素即可。但是还有其他更快的方法')
        print('9.1 最小值和最大值')
        A = [61, 52, 43, 34, 25, 16, 17]
        print('数组A', _deepcopy(A), '中元素的最小的元素为:', self.minimum(A))
        print('可以通过n-1次比较找出一个数组中的上界和下界')
        print('在某些应用中同时找出最小值和最大值')
        print('要设计出一个算法，使之通过渐进最优的Θ(n)次比较，能从n个元素中找出最小值和最大值')
        print('只要独立地找出最小值和最大值，各用n-1次比较，共有2n-2次比较')
        print('事实上，至多3[n/2]次比较久足以同时找出最小值和最大值，做法是记录比较过程中遇到的最大值和最小值')
        print('练习9.1-1 在最坏情况下，利用n+[lgn]-2次比较，即可找到n个元素中的第2小元素')
        print('练习9.1-2 在最坏情况下，同时找到n个数字中的最大值和最小值需要[3n/2]-2次比较')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_2:
    '''
    chpater9.2 note and function
    '''
    def partition(self, A : list, p : int, r : int) -> int:
        '''
        快速排序分堆子过程(并且避免了元素都相同时分堆进入最差情况)
        '''
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

    def randomized_partition(self, A : list, p : int, r : int):
        '''
        快速排序随机分堆子过程
        '''
        i = _randint(p, r)
        A[r], A[i] = A[i], A[r]
        return self.partition(A, p, r)

    def __randomized_select(self, A : list, p : int, r : int, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`
        '''
        assert p <= r      
        if len(A) == 0:
            return None
        if p == r:
            return A[p]
        q = self.randomized_partition(A, p, r)
        k = q - p + 1
        if i == k:
            return A[q]
        elif i < k:
            return self.__randomized_select(A, p, q - 1, i)
        return self.__randomized_select(A, q + 1, r, i - k)

    def randomized_select(self, A : list, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`,利用了`快速排序`分堆的方法(递归调用)
        '''
        assert i <= len(A) and i > 0
        return self.__randomized_select(A, 0, len(A) - 1, i)

    def randomized_select(self, A : list, i : int):
        '''
        解决选择问题的分治算法,期望运行时间为`Θ(n)`,利用了`快速排序`分堆的方法(迭代调用)
        '''
        assert i <= len(A) and i > 0
        if len(A) == 0:
            return None
        return A[i - 1]

    def note(self):
        '''
        Summary
        ====
        Print chapter9.2 note

        Example
        ====
        ```python
        Chapter9_2().note()
        ```
        '''
        print('chapter9.2 note as follow')
        print('9.2 以期望线性时间做选择')
        print('一般选择问题看起来比找最小值的简单选择问题更难。但是，两种问题的渐进运行时间却是相同的：都是Θ(n)')
        print('将介绍一种用来解决选择问题的分治算法，Randomized-select算法，以排序算法为基本模型')
        print('如同排序在快速排序当中一样，此算法的思想也是对输入数组进行递归划分')
        print('但和快速排序不同的是，快速排序会递归处理划分的两边，而Randomized-select只处理划分的一边')
        print('所以快速排序的期望运行时间是Θ(n),而Randomized-select的期望运行时间为Θ(n),证明过程略')
        print('练习9.2-1 在Randomized-select中，对长度为0的数组，不会进行递归调用')
        print('练习9.2-2 指示器随机变量X_k和T(max(k-1,n-k))是独立的')
        print('练习9.2-3 略')
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第1小选择元素为：", self.randomized_select(A, 1))
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第2小选择元素为：", self.randomized_select(A, 2))
        A = [3, 2, 9, 0, 7, 5, 4, 8, 6, 1]
        print('练习9.2-4 数组A', _deepcopy(A), "的第3小选择元素为：", self.randomized_select(A, 3))
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_3:
    '''
    chpater9.3 note and function
    '''
    def partition(self, A : list, p : int, r : int) -> int:
        '''
        快速排序分堆子过程(并且避免了元素都相同时分堆进入最差情况)
        '''
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

    def select(self, A : list, i : int):
        '''
        在一个数组中选择出第i小的元素(当i=1时，即找出最小元素
        '''
        assert i <= len(A)
        return A[i]

    def note(self):
        '''
        Summary
        ====
        Print chapter9.3 note

        Example
        ====
        ```python
        Chapter9_3().note()
        ```
        '''
        print('chapter9.3 note as follow')
        print('9.3 最坏情况线性时间的选择')
        print('现在来看一个最坏情况运行时间为O(n)的选择算法Select')
        print('像9.2中的randomized_select一样，select通过对输入数组的递归划分来找出所求元素')
        print('但是，该算法的基本思想是要保证对数组的划分是个好的划分')
        print('select采用了取自快速排序的确定性划分算法patition并作出了一些修改，把划分主元元素作为其参数')
        print('算法SELECT通过执行下列步骤来确定一个有n>1个元素的输入数组中的第i小的元素。')
        print(' 1.将输入数组的n个元素划分为[n/5]组，每组5个元素，且至多只有一个组由剩下的n mod 5个元素组成')
        print(' 2.寻找[n/5]个组中每一组的中位数，首先对每组中的元素(至多为5个)进行插入排序，然后从排序过的序列中选出中位数')
        print(' 3.对第2步中找出的[n/5]个中位数，递归调用SELECT以找出其下中位数x')
        print(' 4.利用修改过的partition过程，按中位数的中位数x对输入数组进行划分。让k比划分低区的元素数目多1')
        print('  所以x是第k小的元素，并且有n-k个元素在划分的高区')
        print(' 5.如果i=k,则返回x。否则如果i<k,则在低区递归调用SELECT以找出第i小的元素，如果i>k,则在高区找第(i-k)个最小元素')
        print('因此，在[n/5]个组中，除了那个所包含元素可能少于5的组和包含x的那个组之外，至少有一半的组有3个元素大于x')
        print('类似地，小于x的原宿至少有3n/10-6个。因此，在最坏情况下，在第5步中最多有7n/10+6个元素递归调用select')
        print('步骤1,2,4需要O(n)的时间（步骤2对大小为O(1)的集合要调用O(n)次插入排序）')
        print('步骤3花时间T([n/5])，步骤5所需时间至多为T(7n/10+6),假设T是单调递增的')
        print('还需要做如下假设：即任何等于或少于140个元素的输入需要O(1)的时间；这个魔力常数140的起源很快就变得清晰了')
        print('在此假设下，可以得到递归式：')
        print('T(n)=Θ(1), n<=140; T(n)=T([n/5])+T(7n/10+6)+O(n), n>140)')
        print('用定义可以证明T(n)=O(n)')
        print('因此，select的最坏情况运行时间是线性的')
        print('与比较排序一样，select和randomized_select仅仅通过元素间的比较来确定它们之间的相对次序。')
        print('在第8章中，我们知道在比较模型中，即使是在平均情况下，排序仍然需要Ω(nlgn)')
        print('第8章的线性时间排序算法在输入上作了假设。相反地，本章的线性时间选择算法不需要关于输入的任何假设')
        print('它们不受下界Ω(nlgn)的约束，因为没有使用排序就解决了选择问题')
        print('所以本章中选择算法之所以具有线性运行时间，是因为这些算法没有进行排序；线性时间的行为并不是因为对输入做假设所得到的结果')
        print('第8章中的排序算法就是这么做的。在比较模型中，即使是在平均情况下，排序仍然需要Ω(nlgn)的时间')
        print('练习9.3-1 在算法select中，输入元素被分为每组5个元素')
        print(' 如果分成每组3个元素，select无法在线性时间内运行')
        print('练习9.3-2 证明如果n>=140,则至少有[n/4]个元素大于中位数的中位数x，并且至少有[n/4]个元素小于x')
        print('练习9.3-3 怎么让快速排序在最坏情况下以O(nlgn),随机化输入数组多次，使最坏情况发生的概率接近于0,但是运行时间常数项会增加')
        print('练习9.3-4 假设对一个包含有n个元素的集合，某算法只用比较来确定第i小的元素。')
        print(' 证明：无需另外的比较操作，它也能找到比i小的i-1个原宿和比i大的n-i个元素')
        print('练习9.3-5 假设已经有了一个用于求解中位数的黑箱子程序，它在最坏情况下需要线性运行时间。写出一个能解决任意顺序统计量的选择问题的线性时间算法')
        print('练习9.3-6 对于一个含有n个元素的集合来说，所谓k分位数，就是能把已排序的集合分成k个大小相等的集合的k-1个顺序统计量')
        print(' 给出一个能列出某一结合的k分位数的O(nlgk)时间的算法')
        print('练习9.3-7 给出一个O(n)时间的算法，在给定一个有n个不同数字的集合S以及一个正整数k<=n后，', 
            '它能确定出S中最接近其中位数的k个数')
        print('练习9.3-8 设X[1..n]和Y[1..n]为两个数组，每个都包含n个已经排好序的数。', 
            '给出一个求数组X和Y中所有2n个元素的中位数的，O(lgn)时间的算法')
        print('练习9.3-9 最短管道总长和问题')
        print('思考题9-1 已排序的i个最大数：给定一个含有n个元素的集合，希望能用一个基于比较的算法来找出按顺序排列的i个最大元素')
        print('思考题9-2 带权中位数：在O(nlgn)的最坏情况时间内求出n个元素的带权中位数')
        print('思考题9-3 小型顺序统计量:为从n个数字中选出第i个顺序统计量，',
            'SELECT在最坏情况下所使用的比较次数T(n)=Θ(n),但是常数项特别大')
        print('描述一个能用Ui(n)次比较找出n个元素中的第i小元素的算法，其中')
        print('Ui(n)=T(n) i>=n/2; Ui(n)=[n/2]+Ui([n/2])+T(2i)')
        print('证明：如果i<n/2,则Ui(n)=n+O(T(2i)lg(n/i))')
        print('证明：如果i是个小于n/2的常数，则Ui(n)=n+O(lgn)')
        print('证明：如果对k>=2有i=n/k,那么Ui(n)=n+O(T(2n/k)lgk)')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

chapter9_1 = Chapter9_1()
chapter9_2 = Chapter9_2()
chapter9_3 = Chapter9_3()

def printchapter9note():
    '''
    print chapter9 note.
    '''
    print('Run main : single chapter nine!')  
    chapter9_1.note()
    chapter9_2.note()
    chapter9_3.note()

# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
if __name__ == '__main__':  
    printchapter9note()
else:
    pass

```

```py

# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
'''
Class Chapter10_1

Class Chapter10_2

Class Chapter10_3

Class Chapter10_4

'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

import numpy.fft as fft
from numpy import matrix

if __name__ == '__main__':
    import collection as c
else:
    from . import collection as c

class Chapter10_1:
    '''
    chpater10.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter10.1 note

        Example
        ====
        ```python
        Chapter10_1().note()
        ```
        '''
        print('chapter10.1 note as follow')
        print('第三部分 数据结构')
        print('如同在数学中一样，集合也是计算机科学的基础。不过数学中的集合是不变的')
        print('而算法所操作的集合却可以随着时间的改变而增大、缩小或产生其他变化。我们称这种集合是动态的')
        print('接下来的五章要介绍在计算机上表示和操纵又穷动态集合的一些基本技术')
        print('不同的算法可能需要对集合执行不同的操作。例如：许多算法只要求能将元素插入集合、从集合中删除元素以及测试元素是否属于集合')
        print('支持这些操作的动态集合成为字典')
        print('另一些算法需要一些更复杂的操作，例如第6章在介绍堆数据结构时引入了最小优先队列，它支持将一个元素插入一个集合')
        print('1.动态集合的元素')
        print(' 在动态集合的典型实现中，每个元素都由一个对象来表示，如果有指向对象的指针，就可以对对象的各个域进行检查和处理')
        print(' 某些动态集合事件假定所有的关键字都取自一个全序集，例如实数或所有按字母顺序排列的单词组成的集合')
        print(' 全序使我们可以定义集合中的最小元素，或确定比集合中已知元素大的下一个元素等')
        print('2.动态集合上的操作')
        print(' 操作可以分为两类：查询操作和修改操作')
        print(' Search(S, k); Insert(S, x); Delete(S, x); ')
        print(' Minimum(S); Maximum(S); Successor(S, x); Predecessor(S, x)')
        print('第10章 基本数据结构')
        print('很多复杂的数组局够可以用指针来构造，本章只介绍几种基本的结构，包括栈，队列，链表，以及有根树')
        print('10.1 栈和队列')
        print('栈和队列都是动态集合，在这种结构中，可以用delete操作去掉的元素是预先规定好的')
        print('栈实现一种后进先出LIFO的策略；队列实现了先进先出FIFO的策略')
        print('用作栈上的Insert操作称为压入Push，而无参数的Delete操作常称为弹出Pop')
        print('可以用一个数组S[1..n]来实现一个至多有n个元素的栈；数组S有个属性top[S],它指向最近插入的元素')
        print('由S实现的栈包含元素S[1..top[S]],其中S[1]是栈底元素,S[top[S]]是栈顶元素')
        print('当top[S]=0时，栈中不包含任何元素，因而是空的')
        print('要检查一个栈为空可以使用查询操作isEmpty,试图对一个空栈做弹出操作，则称栈下溢')
        print('如果top[S]超过了n，则称栈上溢')
        print('以上三种栈操作的时间均为O(1)')
        s = c.Stack()
        print(s.isEmpty())
        s.push('123')
        s.push(1)
        print(s.count())
        s.pop()
        print(s.pop(), s.count())
        print('队列：Innsert操作称为Enqueue;Delete操作称为Dequeue,队列具有FIFO性质')
        print('队列有头有尾，当元素入队时，将被排在队尾，出队的元素总为队首元素')
        print('用一个数组Q[1..n]实现一个队列，队列具有属性head[Q]指向队列的头，属性tail[Q]指向新元素会被插入的地方')
        print('当元素排列为一个环形时为环形队列，当队列为空时，试图删除一个元素会导致队列的下溢')
        print('当haed[Q]=tail[Q]+1时，队列是满的，这时如果插入一个元素会导致队列的上溢')
        q = c.Queue([1, 2, 3])
        print(q.length())
        q.enqueue(4)
        q.dequeue()
        print(q.array)
        s = c.Stack()
        s.push(4);s.push(1);s.push(3);s.pop();s.push(8);s.pop()      
        print('练习10.1-1: stack:', s.array)
        s = c.TwoStack()
        s.one_push(1)
        s.two_push(9)
        s.one_push(2)
        s.one_push(3)
        s.one_push(4)
        print('练习10.1-2: ')
        print(' ', s.one_all(), s.two_all())
        try:
            s.two_push(8)
        except Exception as err:
            print(' ', err)
        q = c.Queue()
        q.enqueue(4);q.enqueue(1);q.enqueue(3);q.dequeue();q.enqueue(8);q.dequeue()
        print('练习10.1-3: queue:', q.array)
        try:
            q = c.Queue()
            q.dequeue()
        except Exception as err:
            print('练习10.1-4: ')
        dq = c.DoubleQueue()
        dq.enqueue(1);dq.enqueue(2);dq.enqueue_reverse(3)
        print('练习10.1-5: 双端队列：', dq.array)
        q = c.QueueUsingStack()
        q.enqueue(1);q.enqueue(2);q.enqueue(3)
        print('练习10.1-6: ', q.dequeue(), q.dequeue(), q.dequeue())
        s = c.StackUsingQueue()
        s.push(1);s.push(2);s.push(3)
        print('练习10.1-7: ', s.pop(), s.pop(), s.pop())   
        # python src/chapter10/chapter10note.py
        # python3 src/chapter10/chapter10note.py
        
class Chapter10_2:
    '''
    chpater10.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter10.2 note

        Example
        ====
        ```python
        Chapter10_2().note()
        ```
        '''
        print('chapter10.2 note as follow')
        print('10.2 链表')
        print('在链表这种数据结构中，各对象按照线性顺序排序')
        print('链表与数组不同,数组的线性序是由数组的下标决定的，而链表中的顺序是由各对象中的指针决定的')
        print('链表可以用来简单而灵活地表示动态集合，但效率可能不一定很高')
        print('双链表L的每一个元素都是一个对象，每个对象包含一个关键字域和两个指针域：next和prev,也可以包含一些其他的卫星数据')
        print('对链表中的某个元素x，next[x]指向链表中x的后继元素，而prev[x]则指向链表中x的前驱元素。')
        print('如果prev[x]=NIL,则元素x没有前驱结点，即它是链表的第一个元素，也就是头(head);')
        print('如果next(x)=NIL,则元素x没有后继结点，即它是链表的最后一个元素，也就是尾')
        print('属性head[L]指向表的第一个元素。如果head[L]=NIL,则该链表为空')
        print('一个链表可以呈现为好几种形式。它可以是单链接的或双链接的，已排序的或未排序的，环形的或非环形的')
        print('在本节余下的部分，假定所处理的链表都是无序的和双向链接的')
        print('链表的搜索操作：简单的线性查找方法')
        print('链表的插入：给定一个已经设置了关键字的新元素x，过程LIST-INSERT将x插到链表的前端')
        print('链表的删除：从链表L中删除一个元素x，它需要指向x的指针作为参数')
        print('但是，如果希望删除一个具有给定关键字的元素，则要先调用LIST-SEARCH过程，', 
            '因而在最坏情况下的时间为Θ(n)')
        print('哨兵(sentinel)是个哑(dummy)对象，可以简化边界条件')
        l = c.List()
        l.insert(1);l.insert(2);l.insert(3)     
        print('链表中的元素总和:', l.all(), l.head.value, 
            l.head.next.value, l.head.next.next.value, l.count())
        l.delete_bykey(0)
        print(l.all())
        l.delete_bykey(2)
        print(l.all())
        print('练习10.2-1: 动态集合上的操作INSERT能用一个单链表在O(1)时间内实现')
        s = c.StackUsingList()
        s.push(1);s.push(2);s.push(3);s.pop()
        print('练习10.2-2: ', s.all())
        q = c.QueueUsingList()
        q.enqueue(1);q.enqueue(2);q.enqueue(3);q.dequeue()
        print('练习10.2-3: ', q.all())
        print('练习10.2-4: 不用哨兵NIL就可以了')
        print('练习10.2-5: 用环形单链表来实现字典操作INSERT,DELETE和SEARCH，以及运行时间')
        print('练习10.2-6: 应该选用一种合适的表数据结构，以便之处在O(1)时间内的Union操作')
        print('练习10.2-7: 链表反转过程Θ(n)的非递归过程，对含有n个元素的单链表的链进行逆转')
        print(' 除了链表本身占用的空间外，该过程仅适用固定量的存储空间')
        print('练习10.2-8: 如何对每个元素仅用一个指针np[x](而不是两个指针next和prev)来实现双链表')
        print(' 假设所有指针值都是k位整型数，且定义np[x] = next[x] XOR prev[x],即next[x]和')
        print(' prev[x]的k位异或(Nil用0表示)。注意要说明访问表头所需的信息，以及如何实现在该表上的SEARCH,INSERT和DELETE操作')
        print(' 如何在O(1)时间内实现这样的表')
        # python src/chapter10/chapter10note.py
        # python3 src/chapter10/chapter10note.py

class Chapter10_3:
    '''
    chpater10.3 note and function
    '''

    free = None
    node = c.ListNode()
    
    def allocate_object(self):
        if self.free == None:
            raise Exception('Exception: out of space')
        else:
            self.node = self.free
            self.free = self.node.next
            return node

    def free_object(self, x : c.ListNode):
        x.next = self.free
        self.free = x

    def note(self):
        '''
        Summary
        ====
        Print chapter10.3 note

        Example
        ====
        ```python
        Chapter10_3().note()
        ```
        '''
        print('chapter10.3 note as follow')
        print('指针和对象的实现')
        print('有些语言(如FORTRAN)中不提供指针与对象数据类型')
        print('对象的多重数组表示')
        print(' 对一组具有相同域的对象，每一个域都可以用一个数组表示')
        print(' 动态结合现有的关键字存储在数组key，而指针存储在数组next和prev中')
        print(' 对于某一给定的数组下标x, key[x], next[x], prev[x]就共同表示链表中的一个对象')
        print(' 在这种解释下，一个指针x即为指向数组key, next, prev的共同下标')
        print(' 在给出的伪代码中，方括号既可以表示数组的下标，又可以表示对象的某个域(属性)')
        print(' 无论如何，key[x],next[x],prev[x]的含义都与实现是一致的')
        print('对象的单数组表示')
        print(' 计算机存储器中的字是用整数0到M-1来寻址的，此处M是个足够大的整数。')
        print(' 在许多程序设计语言中，一个对象占据存储中的一组连续位置，指针即指向某对象所占存储区的第一个位置，后续位置可以通过加上相应的偏移量进行寻址')
        print(' 对不提供显式指针数据类型的程序设计环境,可以采取同样的策略来实现对象。')
        print(' 一个对象占用一个连续的子数组A[j..k]。对象的每一个域对应着0到k-j之间的一个偏移量，而对象的指针是下标j')
        print(' key,next,prev对应的偏移量为0、1和2。给定指针i，为了读prev[i],将指针值i与偏移量2相加,即读A[i+2]')
        print(' 这种单数组表示比较灵活，它允许在同一数组中存放不同长度的对象。')
        print(' 要操纵一组异构对象要比操纵一组同构对象(各对象具有相同的域)更困难')
        print(' 因为考虑的大多数数据结构都是由同构元素所组成的，故用多重数组表示就可以了')
        print('分配和释放对象')
        print(' 为向一个用双链表表示的动态集合中插入一个关键字，需要分配一个指向链表表示中当前未被利用的对象的指针')
        print(' 在某些系统中，是用废料收集器来确定哪些对象是未用的')
        print(' 假设多重数组表示中数组长度为m，且在某一时刻，动态数组包含n<=m个元素。')
        print(' 这样,n个对象即表示目前在动态集合中的元素，而另m-n个元素是自由的，它们可以用来表示将要插入动态集合中的元素')
        print(' 把自由对象安排成一个单链表，称为自由表。自由表仅用到next数组，其中存放着表中的next指针。')
        print(' 该自由表的头被置于全局变量free中。当链表L表示的动态集合非空时，自由表将与表L交错在一起')
        print('自由表是一个栈：下一个分配的对象是最近被释放的那个。可以用栈操作PUSH和POP的表实现方式来分别实现对象的分配和去分配过程。')
        print('假设全局变量free指向自由表的第一个元素')
        l = c.List()
        l.insert(13);l.insert(4);l.insert(8);l.insert(19);l.insert(5);l.insert(11);
        print('练习10.3-1: 序列[13,4,8,19,5,11]的单链表所有元素的表示为：', l.all())
        print('练习10.3-2: 用一组用单数组表示实现的同构对象，写出其过程ALLOCATE-OBJECT和FREE-OBJECT')
        print('练习10.3-3: 在过程ALLOCATE-OBJECT和FREE-OBJECT的实现中，不需要置或重置对象的prev域')
        print('练习10.3-4: 希望一个双链表中的所有元素在存储器中能够紧凑地排列在一起，例如使用多重数组表示中的前m下标位置')
        print('练习10.3-5: 设L是一个长度为m的双链表，存储在长度为n的数组key、next和prev中。')
        print(' 结社这些数组由维护双链自由表F的两个过程ALLOCATE-OBJECT和FREE-OBJECT')
        # python src/chapter10/chapter10note.py
        # python3 src/chapter10/chapter10note.py

class Chapter10_4:
    '''
    chpater10.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter10.4 note

        Example
        ====
        ```python
        Chapter10_4().note()
        ```
        '''
        print('chapter10.4 note as follow')
        print('10.4 有根树的表示')
        print('前一节中链表的表示方法可以推广至任意同构的数据结构上。用链接数据结构表示有根树')
        print('首先讨论二叉树，然后提出一种适用于结点子女数任意的有根树表示方法')
        print('二叉树')
        print(' 用域p,left,right来存放指向二叉树T中的父亲，左儿子和右儿子的指针')
        print(' 如果P[x]=NIL,则x为根。如果结点无左儿子，则left[x]=NIL,对右儿子也类似。')
        print(' 整个树T的根由属性root[T]指向。如果root[T]=NIL,则树为空')
        print('分支数无限制的有根树')
        print('上面二叉树的表示方法可以推广至每个结点的子女数至多为常数k的任意种类树；')
        print('用child_1,child_2,...,child_k来取代left和right域。')
        print('如果树种结点的子女数是无限制的，那么这种方法就不适用了，')
        print('此外，即使结点的子女数k以一个很大的常数为界，但多数结点只有少量子女，则会浪费大量的存储空间')
        print('可以用二叉树很方便地表示具有任意子女数的树。')
        print('这种方法的优点是对任意含n个结点的有根树仅用O(n)的空间')
        print('树的其他表示：有时，可以用另外一些方法来表示有根树。', 
            '例如在第六章中，用一个数组加上下标的形式来表示基于完全二叉树的堆')
        print('将在第21章中出现的树可只由叶向根的方向遍历，故只用到父指针，而没有指向子女的指针')
        print('练习10.4-1 下列域表示的，根在下标6处的二叉树')
        print(' 索引为 6 1 4 7 3 5 9')
        print(' 键值为18 12 10 7 4 2 21')
        btree = c.BinaryTree()    
        btree.addnode(None, None, 7, 7)
        btree.addnode(10, None, 3, 4)
        btree.addnode(None, None, 5, 2)
        btree.addnode(None, None, 9, 21)
        btree.addnode(7, 3, 1, 12)
        btree.addnode(5, 9, 4, 10)
        btree.addnode(1, 4, 6, 18)
        btree.renewall()
        print('练习10.4-2 请写出一个O(n)时间的递归过程，在给定含n个结点的二叉树后，它可以将树中每个结点的关键字输出来')
        print(' 递归过程（还必须找出根节点在哪里）所有节点的索引和键值为：', btree.findleftrightnode(btree.lastnode))
        print('练习10.4-3 请写出一个O(n)时间的非递归过程，将给定的n结点二叉树中每个结点的关键字输出出来。可以利用栈作为辅助数据结构')
        print(' 非递归过程所有节点的索引和键值为：', btree.all())
        print('练习10.4-4 对于任意的用左孩子，右兄弟表示存储的，含n个结点的有根树，写出一个O(n)时间过程来输出每个结点的关键字')
        print(' 所有键值集合:', btree.keys())
        print('练习10.4-5 写出一个O(n)时间的非递归过程，输出给定的含n个结点的二叉树中每个结点的关键字')
        print('练习10.4-6 在任意有根树的每个左儿子，右儿子都有三个指针left-child,right-child,parent')
        print(' 从任意结点出发，都可以在常数时间到达其父亲结点；可以在与子女数成线性关系的时间到达其孩子')
        print(' 并且只利用两个指针和一个布尔值')
        print('思考题10-1 链表之间的比较:对下表中的四种列表，每一种动态集合操作的渐进最坏情况运行时间是什么')
        print(' 未排序的单链表，已排序的单链表，未排序的双链表，已排序的双链表')
        print(' SEARCH(L,k),INSERT(L,x),DELETE(L,x),SUCCESSOR(L,x),PREDECESSOR(L,x)')
        print(' MINIMUM(L),MAXIMUM(L)')
        print('思考题10-2 用链表实现的可合并堆')
        print(' 一个可合并堆支持这样几种操作：MAKE-HEAP(创建一个空的可合并堆)，INSERT,MINIMUM,EXTRACT-MIN和UNION')
        print('思考题10-3 在已排序的紧凑链表中搜索')
        print(' 在一个数组的前n个位置中紧凑地维护一个含n个元素的表。假设所有关键字均不相同，且紧凑表是排序的')
        print(' 若next[i]!=None,有key[i]<key[next[i]],在这些假设下，试说明如下算法能在O(sqrt(n))期望时间内完成链表搜索')
        l = c.List()
        l.insert(1);l.insert(2);l.insert(3);l.insert(4);l.insert(5);l.insert(6);
        print('key为3的链表节点为：', l.compact_search(3))
        print('key为4的链表节点为：', l.compact_search(4))
        print('key为3的链表节点为：', l.compact_list_search(1, 6))
        print('key为4的链表节点为：', l.compact_list_search(2, 6))
        # python src/chapter10/chapter10note.py
        # python3 src/chapter10/chapter10note.py

chapter10_1 = Chapter10_1()
chapter10_2 = Chapter10_2()
chapter10_3 = Chapter10_3()
chapter10_4 = Chapter10_4()

def printchapter10note():
    '''
    print chapter10 note.
    '''
    print('Run main : single chapter ten!')  
    chapter10_1.note()
    chapter10_2.note()
    chapter10_3.note()
    chapter10_4.note()

# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
if __name__ == '__main__':  
    printchapter10note()
else:
    pass

```

```py

from __future__ import division, absolute_import, print_function
import json as _json
from random import randint as _randint
class Stack:
    '''
    栈
    '''
    def __init__(self, iterable = None):
        self.__top = -1
        self.array = []
        if iterable != None:
            self.array = list(iterable)
    
    def isEmpty(self): 
        '''
        栈是否为空

        Return
        ===
        `isempty` -> bool
        '''
        return self.__top == -1

    def push(self, item):
        '''
        入栈操作
        '''
        self.__top = self.__top + 1
        self.array.append(item)

    def pop(self):
        '''
        出栈操作
        '''
        if self.isEmpty() == True:
            raise Exception('the stack has been empty')
        else:
            self.__top = self.__top - 1
            return self.array.pop()

    def count(self):
        '''
        返回栈中所有元素的总数
        '''
        return len(self.array)

class TwoStack:
    '''
    用一个数组实现的两个栈
    '''
    def __init__(self, size = 5):
        self.__one_top = -1
        self.__two_top = size
        self.__size = size
        self.__array = list(range(size))    
    
    def one_push(self, item):
        self.__judgeisfull()
        self.__one_top += 1
        self.__array[self.__one_top] = item

    def one_pop(self):
        self.__judgeisempty()
        x = self.__array[self.__one_top]
        self.__one_top -= 1
        return x

    def two_push(self, item):
        self.__judgeisfull()
        self.__two_top -= 1
        self.__array[self.__two_top] = item

    def two_pop(self):
        self.__judgeisempty()
        x = self.__array(self.__two_top)
        self.__two_top += 1
        return x

    def one_all(self):
        array = []
        if self.__one_top != -1:
            for i in range(self.__one_top):
                array.append(self.__array[i])
        return array

    def two_all(self):
        array = []
        if self.__two_top != self.__size:
            for i in range(self.__two_top, self.__size):
                index = self.__size + self.__two_top - i - 1
                array.append(self.__array[index])
        return array

    def __judgeisfull(self):
        if self.__one_top + 1 == self.__two_top:
            raise Exception('Exception: stack is full!')

    def __judgeisempty(self):
        if self.__one_top == -1 or self.__two_top == self.__size:
            raise Exception('stack is full!')
class StackUsingQueue:
    '''
    用队列实现的栈
    '''
    def __init__(self, iterable = None):
        self.__queue1 = Queue()
        self.__queue2 = Queue()
    
    def push(self, item):
        self.__queue1.enqueue(item)

    def pop(self):
        for i in range(self.__queue1.length() - 1):
            self.__queue2.enqueue(self.__queue1.dequeue())
        x = self.__queue1.dequeue()
        for i in range(self.__queue2.length()):
            self.__queue1.enqueue(self.__queue2.dequeue())
        return x
        
    def count(self):
        return self.__queue1.length()
        
class Queue:
    '''
    队列
    '''
    def __init__(self, iterable = None):
        self.tail = 0
        self.array = []
        if iterable != None:
            self.array = list(iterable)

    def enqueue(self, item):
        '''
        元素`item`加入队列
        '''
        self.array.append(item)
        if self.tail == self.length:
            self.tail = 0
        else:
            self.tail = self.tail + 1

    def dequeue(self):
        '''
        元素出队列
        '''
        if self.length() == 0:
            raise Exception('Exception: the queue has been empty')
        x = self.array[0]
        self.array.remove(x)
        return x

    def length(self):
        return len(self.array)
class DoubleQueue:
    '''
    双向队列
    '''
    def __init__(self, iterable = None):
        self.tail = 0
        self.array = []
        if iterable != None:
            self.array = list(iterable)

    def enqueue(self, item):
        '''
        元素`item`加入队列
        '''
        self.array.append(item)
        if self.tail == self.length:
            self.tail = 0
        else:
            self.tail = self.tail + 1

    def dequeue(self):
        '''
        元素出队列
        '''
        if self.length() == 0:
            raise Exception('Exception: the queue has been empty')
        x = self.array[0]
        self.array.remove(x)
        return x

    def enqueue_reverse(self, item):
        self.array.insert(0, item)

    def dequeue_reverse(self):
        self.array.pop()

    def length(self):
        return len(self.array)

class QueueUsingStack:
    '''
    用栈实现的队列
    '''
    def __init__(self, iterable = None):
        self.__stack1 = Stack()
        self.__stack2 = Stack()
    
    def enqueue(self, item):
        self.__stack1.push(item)

    def dequeue(self):
        for i in range(self.__stack1.count() - 1):
            self.__stack2.push(self.__stack1.pop())
        x = self.__stack1.pop()
        for i in range(self.__stack2.count()):
            self.__stack1.push(self.__stack2.pop())
        return x
    
    def count(self):
        return self.__stack1.count()

class ListNode:
    '''
    链表节点
    '''
    def __init__(self, value = None):
        '''
        链表节点
        ```python
        >>> ListNode() 空节点   
        >>> ListNode(value) 值为value的链表节点
        ```
        '''
        self.value = value
        self.key = -1      
        self.prev = None
        self.next = None

    def __str__(self):
        return "key:" + str(self.key) + ";value:" + str(self.value)

    def getisNone(self):
        '''
        链表节点是否为空
        '''
        return self.key == None

    isNone = property(getisNone, None)

class List:   
    '''
    链表
    '''    
    def __init__(self):
        '''
        初始化一个空链表
        '''
        self.head = None
        self.tail = None
        self.next = None
        self.__length = 0

    def search(self, k):
        '''
        找出键值为k的链表节点元素，最坏情况为`Θ(n)`
        '''
        x = self.head
        while x.value != None and x.key != k:
            x = x.next
        return x

    def get_random_node(self) -> ListNode:
        num = _randint(0, self.count() - 1)
        j = self.head
        for iterate in range(num):
            j = j.next
        return j

    def compact_search(self, k):
        '''
        已经排序的链表中找出键值为k的链表节点元素，期望情况为`O(sqrt(n))`

        Args
        ===
        `k` : 待寻找元素的键值

        '''
        n = self.count()
        i = self.head
        while i != None and i.key > k:
            num = _randint(0, n - 1)
            j = self.get_random_node()
            if i.key < j.key and j.key <= k:
                i = j
                if i.key == k:
                    return i
            i = i.next
        if i == None or i.key < k:
            return None
        else:
            return i

    def compact_list_search(self, k, t):
        '''
        已经排序的链表中找出键值为k的链表节点元素，期望情况为`O(sqrt(n))`

        Args
        ===
        `k` : 待寻找元素的键值

        `t` : 循环迭代次数上界

        '''
        i = self.head
        for q in range(t):
            j = self.get_random_node()
            if i.key > j.key:
                i = j
                if i.key == k:
                    return i
        while i != None and i.key < k:
            i = i.next
        if i != None and i.key > k:
            return None
        else:
            return i

    def findtail(self):
        x = self.head
        while x != None and x.value != None:
            prev = x
            x = x.next
        return prev

    def insert(self, x):
        '''
        链表插入元素x
        '''
        self.__insert(ListNode(x))

    def __insert(self, x : ListNode):
        # 插入的元素按增量键值去
        x.key = self.__length;   
        # 把上一个头节点放到下一个节点去   
        x.next = self.head
        # 判断是否第一次插入元素
        if self.head != None and self.head.isNone == False:
            self.head.prev = x
        # 新插入的元素放到头节点去
        self.head = x
        # 新插入的节点前面没有元素
        x.prev = None
        self.__increse_length()

    def delete(self, item, key):
        '''
        链表删除元素x
        '''
        if type(item) is not ListNode:
            x = ListNode(item)
            x.key = key
        else:
            x = item
        if x.prev != None and x.prev.isNone == False:
            x.prev.next = x.next
        else:
            self.head = x.next
        if x.next != None and x.next.isNone == False:
            x.next.prev = x.prev
        self.__length -= 1

    def delete_bykey(self, k : int) -> ListNode:
        '''
        根据键值删除元素
        '''
        x = self.search(k)
        self.delete(x, x.key)
        return x.value

    def count(self):
        '''
        返回链表中元素的数量总和
        '''
        return self.__length

    def all(self):
        '''
        返回链表中所有元素组成的集合
        '''
        array = []
        x = self.head
        count = self.count()
        while x != None:
            value = x.value
            if value != None:
                array.append(value)
            x = x.next
        array.reverse()
        return array

    def __increse_length(self):
        self.__length += 1       

    def __reduce_length(self):
        self.__length -= 1

class QueueUsingList:
    '''
    使用链表构造的队列
    '''
    def __init__(self):
        self.__list = List()
        self.__length = 0

    def enqueue(self, item):
        self.__list.insert(item)
        self.__length += 1

    def dequeue(self):
        x = self.__list.findtail()
        self.__list.delete(x, x.key)
        self.__length -= 1
        return x.value

    def count(self):
        self.__length()

    def all(self):
        return self.__list.all()

class StackUsingList:
    '''
    使用链表构造的栈
    '''
    def __init__(self):
        self.__list = List()
        self.__length = 0

    def push(self, item):
        self.__list.insert(item)
        self.__length += 1

    def pop(self):
        x = self.__list.head
        self.__list.delete(x, x.key)
        self.__length -= 1
        return x.value

    def count(self):
        self.__length()

    def all(self):
        return self.__list.all()

class BTreeNode:
    '''
    二叉树结点
    '''
    def __init__(self, left, right, index, \
            key , leftindex, rightindex):
        '''

        二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        '''
        self.leftindex = leftindex
        self.rightindex = rightindex
        self.left = left
        self.right = right
        self.index = index
        self.key = key

class BinaryTree:
    '''
    二叉树
    '''
    def __init__(self):
        '''
        二叉树
        '''
        self.lastnode = None
        self.root = None
        self.nodes = []

    def addnode(self, leftindex : int, rightindex : int, selfindex : int, selfkey):
        '''
        加入二叉树结点

        Args
        ===
        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        `selfindex` : 结点自身索引值

        `selfkey` : 结点自身键值

        '''
        leftnode = self.findnode(leftindex)
        rightnode = self.findnode(rightindex)
        x = BTreeNode(leftnode, rightnode, selfindex, \
            selfkey, leftindex, rightindex)
        self.nodes.append(x)
        self.lastnode = x
        return x
        
    def renewall(self) -> None:
        '''
        更新/连接/构造二叉树
        '''
        for node in self.nodes:
            node.left = self.findnode(node.leftindex)
            node.right = self.findnode(node.rightindex)
    
    def findleftrightnode(self, node : BTreeNode) -> list:
        '''
        找出二叉树某结点的所有子结点

        Args
        ===
        `node` : BTreeNode : 某结点
        '''
        array = []
        if node != None:
            # 递归找到左儿子所有的结点
            leftnodes = self.findleftrightnode(node.left)
            # 递归找到右兄弟所有的结点
            rightnodes = self.findleftrightnode(node.right)
            if leftnodes != None and len(leftnodes) != 0:
                # 连接两个集合
                array = array + leftnodes
            if rightnodes != None and len(rightnodes) != 0:
                # 连接两个集合
                array = array + rightnodes
            # 将自己本身的结点也加入集合
            array.append({ "index":node.index, "key" : node.key})
            if len(array) == 0:
                return None
            return array
        return None

    def all(self) -> list:
        '''
        返回二叉树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def keys(self) -> list:
        '''
        返回二叉树中所有结点键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append(node.key)
        return array

    def findnode(self, index : int):
        '''
        根据索引寻找结点`O(n)`

        Args
        ===
        `index` : 索引值
        '''
        if index == None:
            return None
        for node in self.nodes:
            if node.index == index:
                return node
        return None

```

```py

# python src/chapter11/chapter11note.py
# python3 src/chapter11/chapter11note.py
'''
Class Chapter11_1

Class Chapter11_2

Class Chapter11_3

Class Chapter11_4

Class Chapter11_5

'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter11_1:
    '''
    chpater11.1 note and function
    '''

    def ELFhash(self, key : str, mod):
        h = 0
        for c in key:
            h = (h << 4) + ord(c)
            g = h & 0xF0000000
            if g != 0:
                h ^= g >> 24;
            h &= ~g;  
        return h // mod  

    def note(self):
        '''
        Summary
        ====
        Print chapter11.1 note

        Example
        ====
        ```python
        Chapter11_1().note()
        ```
        '''
        print('chapter11.1 note as follow')
        print('第11章 散列表')
        print('散列表(hash table 哈希表)，是根据关键码值(key value)而直接进行访问的数据结构')
        print(' 通过把关键码值映射到表中一个位置来访问记录，以加快查找的速度')
        print(' 这个函数叫做散列函数，存放记录的数组叫散列表')
        print('对不同的关键字可能得到同一散列地址，即k1≠k2，而f(k1)=f(k2)，这种现象称为碰撞', 
            '（英语：Collision）。具有相同函数值的关键字对该散列函数来说称做同义词。', 
            '综上所述，根据散列函数f(k)和处理碰撞的方法将一组关键字映射到一个有限的连续的地址集', 
            '（区间）上，并以关键字在地址集中的“像”作为记录在表中的存储位置，这种表便称为散列表，', 
            '这一映射过程称为散列造表或散列，所得的存储位置称散列地址。')
        print('字符串dugu的一个哈希值为：', self.ELFhash('dugu', 31))
        print('在很多应用中，都要用到一种动态集合结构，它仅支持INSERT,SEARCH的DELETE字典操作')
        print('实现字典的一种有效数据结构为散列表(HashTable)')
        print('在最坏情况下，在散列表中，查找一个元素的时间在与链表中查找一个元素的时间相同')
        print('在最坏情况下都是Θ(n)')
        print('但在实践中，散列技术的效率是很高的。在一些合理的假设下，在散列表中查找一个元素的期望时间为O(1)')
        print('散列表是普通数组概念的推广，因为可以对数组进行直接寻址，故可以在O(1)时间内访问数组的任意元素')
        print('如果存储空间允许，可以提供一个数组，为每个可能的关键字保留一个位置，就可以应用直接寻址技术')
        print('当实际存储的关键字数比可能的关键字总数较小时，这是采用散列表就会较直接数组寻址更为有效')
        print('在散列表中，不是直接把关键字用作数组下标，而是根据关键字计算出下标。')
        print('11.2着重介绍解决碰撞的链接技术。')
        print('所谓碰撞，就是指多个关键字映射到同一个数组下标位置')
        print('11.3介绍如何利用散列函数，根据关键字计算出数组的下标。')
        print('11.4介绍开放寻址法，它是处理碰撞的另一种方法。散列是一种极其有效和实用的技术，基本的字典操作只需要O(1)的平均时间')
        print('11.5解释当待排序的关键字集合是静态的，\"完全散列\"如何能够在O(1)最坏情况时间内支持关键字查找')
        print('11.1 直接寻址表')
        print('当关键字的全域U比较小时，直接寻址是一种简单而有效的技术。假设某应用要用到一个动态集合')
        print('其中每个元素都有一个取自全域U的关键字，此处m是一个不很大的数。另外假设没有两个元素具有相同的关键字')
        print('为表示动态集合，用一个数组(或直接寻址表)T[0...m-1],其中每个位置(或称槽)对应全域U中的一个关键字')
        print('对于某些应用，动态集合中的元素可以放在直接寻址表中。亦即不把每个元素的关键字及其卫星数据都放在直接寻址表外部的一个对象中')
        print('但是，如果不存储关键字，就必须有某种办法来确定某个槽是否为空')
        print('练习11.1-1: 考虑一个由长度为m的直接寻址表T表示的动态集合S。给出一个查找S的最大元素的算法过程')
        print(' 所给的过程在最坏情况下的运行时间是O(m)')
        print('练习11.1-2: 位向量(bit vector)是一种仅包含0和1的数组。长度为m的位向量所占空间要比包含m个指针的数组少得多')
        print(' 请说明如何用一个位向量来表示一个包含不同元素的动态集合。字典操作的运行时间应该是O(1)')
        print('练习11.1-3: 说明如何实现一个直接寻址表，使各元素的关键字不必都相同，且各元素可以有卫星数据。')
        print('练习11.1-4: 希望通过利用一个非常大的数组上直接寻址的方式来实现字典')
        print(' 开始时，该数组中可能包含废料，但要对整个数组进行初始化是不实际的，因为该组的规模太大')
        print(' 请给出在大数组上实现直接寻址字典的方案。每个存储的对象占用O(1)空间')
        print(' 操作SEARCH,INSERT和DELETE的时间为O(1),对数据结构初始化的时间O(1)')
        print(' 可以利用另外一个栈，其大小等于实际存储在字典中的关键字数目，以帮助确定大型数组中某个给定的项是否是有效的')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_2:
    '''
    chpater11.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.2 note

        Example
        ====
        ```python
        Chapter11_2().note()
        ```
        '''
        print('chapter11.2 note as follow')
        print('11.2 散列表')
        print('直接寻址技术存在着一个明显的问题：如果域U很大，',
            '在一台典型计算机的可用内存容量限制下，要在机器中存储大小为U的一张表T就有点不实际甚至是不可能的了')
        print('实际要存储的关键字集合K相对于U来说可能很小，因而分配给T的大部分空间都要浪费掉')
        print('当存储在字典中的关键字集合K比所有可能的关键字域U要小的多时，散列表需要的存储空间要比直接寻址少很多')
        print('特别地，在保持仅需O(1)时间即可在散列表中查找一个元素的好处情况下，存储要求可以降至Θ(|K|)')
        print('在直接寻址方式下，具有关键字k的元素被存放在槽k中。在散列方式下，该元素处于h(k)中')
        print('亦即，利用散列函数h,根据关键字k计算出槽的位置。函数h将关键字域U映射懂啊散列表T[0..m-1]的槽位上：')
        print('这时，可以说一个具有关键字k到元素是被散列在槽h(k)上，或说h(k)是关键字k的散列值')
        print('两个关键字可能映射到同一个槽上。将这种情形称为发生了碰撞')
        print('当然，最理想的解决方法是完全避免碰撞')
        print('可以考虑选用合适的散列函数h。在选择时有一个主导思想，就是使h尽可能地\"随机\",从而避免或至少最小化碰撞')
        print('当然，一个散列函数h必须是确定的，即某一给定的输入k应始终产生相同的结果h(k),')
        print('通过链接法解决碰撞')
        print(' 在链接法中，把散列到同一槽中的所有元素都放在同一个链表中，槽j中有一个指针，它指向由所有散列到j的元素构成的链表的头')
        print(' 插入操作的最坏情况运行时间为O(1).插入过程要快一些，因为假设要插入的元素x没有出现在表中；如果需要，在插入前执行搜索，可以检查这个假设(付出额外代价)')
        print('CHAINED-HASH-INSERT(T, x)')
        print(' insert x at the head of list T[h(key[x])]')
        print('CHAINED-HASH-SEARCH(T, x)')
        print(' search for an element with k in list T[h(k)]')
        print('CHAINED-HASH-DELETE(T, x)')
        print(' delete x from the list T[h(key[x])]')
        print('对用链接法散列的分析')
        print(' 采用链接法后散列的性能怎样呢？特别地，要查找一个具有给定关键字的原宿需要多长时间呢？')
        print(' 给定一个能存放n个元素的，具有m个槽位的散列表T，定义T的装载因子a为n/m,即一个链表中平均存储的元素数')
        print(' 分析以a来表达，a可以小于、等于或者大于1')
        print('用链接法散列的最坏情况性能很差；所有的n个关键字都散列到同一个槽中，从而产生出一个长度为n的链表')
        print('最坏情况下查找的时间为Θ(n),再加上计算散列函数的时间，这么一来就和用一个链表来来链接所有的元素差不多了。显然，')
        print('散列方法的平均性态依赖于所选取的散列函数h在一般情况下，将所有的关键字分布在m个槽位上的均匀程度')
        print('先假定任何元素散列到m个槽中每一个的可能性是相同的，且与其他元素已被散列到什么位置上是独立无关的')
        print('称这个假设为简单一致散列')
        print('假定可以在O(1)时间内计算出散列值h(k),从而查找具有关键字为k的元素的时间线性地依赖于表T[h(k)]的长度为n')
        print('先不考虑计算散列函数和寻址槽h(k)的O(1)时间')
        print('定理11.1 对一个用链接技术来解决碰撞的散列表，在简单一致散列的假设下，一次不成功查找期望时间为Θ(1+a)')
        print('定理11.2 在简单一致散列的假设下，对于用链接技术解决碰撞的散列表，平均情况下一次成功的查找需要Θ(1+a)时间')
        print('练习11.2-1: 假设用一个散列函数h，将n个不同的关键字散列到一个长度为m的数组T中。')
        print(' 假定采用的是简单一致散列法，那么期望的碰撞数是多少？')
        print('练习11.2-2: 对于一个利用链接法解决碰撞的散列表，说明将关键字5,28,19,15,20,33,12,17,10')
        print(' 设该表中有9个槽位，并设散列函数为h(k)=k mod 9')
        print('练习11.2-3: 如果将链接模式改动一下，使得每个链表都能保持已排序顺序，散列的性能就可以有很大的提高。')
        print(' 这样的改动对成功查找、不成功查找、插入和删除操作的运行时间有什么影响')
        print('练习11.2-4: 在散列表内部，如何通过将所有未占用的槽位链接成一个自由链表，来分配和去分配元素的存储空间')
        print(' 假定一个槽位可以存储一个标志、一个元素加上一个或两个指针')
        print(' 所有的字典和自由链表操作应具有O(1)的期望运行时间')
        print('练习11.2-5: 有一个U的大小为n的子集，它包含了均散列到同一个槽位中的关键字，这样对于带链接的散列表，最坏情况下查找时间为Θ(n)')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_3:
    '''
    chpater11.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.3 note

        Example
        ====
        ```python
        Chapter11_3().note()
        ```
        '''
        print('chapter11.3 note as follow')
        print('11.3 散列函数')
        print('好的散列函数的特点')
        print('一个好的散列函数应近似地满足简单一致散列的假设：每个关键字都等可能地散列到m个槽位的任何一个之中去')
        print('并与其它的关键字已被散列到哪一个槽位中无关')
        print('不幸的是：一般情况下不太可能检查这一条件是否成立，因为人们很少可能知道关键字所符合的概率分布，而各关键字可能并不是完全互相独立的')
        print('有时也能知道关键字的概率分布。例如：已知各关键字都是随机的实数k，独立地、一致地分布于范围[0,1)')
        print('在实践中，常常可以运用启发式技术来构造好的散列函数')
        print('例如，在一个编译器的符号表中，关键字都是字符串，表示程序中的标识符')
        print('同一个程序中，经常会出现一些很相近的符号，如pt和pts。')
        print('一个好的散列函数应能最小化将这些相近符号散列到同一个槽中的可能性')
        print('\"除法散列\"用一个特定的质数来除所给的关键字，所得的余数即为该关键字的散列值')
        print('假定所选择的质数与关键字分布中的任何模式都是无关的，这种方法常常可以给出很好的结果')
        print('散列函数的某些应用可能会要求比简单一致散列更强的性质，例如可能希望某些很近似的关键字具有截然不同的散列值')
        print('将关键字解释为自然数')
        print(' 如果所给关键字不是自然数，则必须有一种方法来将它们解释为自然数')
        print(' 标识符pt可以被解释为十进制整数对(112,116),pt即为(112*128)+116=14452')
        print('11.3.1 除法散列法')
        print(' 通过取k除以m的余数，来将关键字k映射到m个槽的摸一个中去，亦即散列函数为h(k) = k mod m')
        print(' 例如，如果散列表的大小为m=12,所给关键字为k=100,则h(k)=4。这种方法只要一次除法操作，所以比较快')
        print(' 应用除法散列时，要注意m的选择，m不应是2的幂；可以选作m的值常常是与2的整数幂不太接近的质数')
        print(' 例如，假设我们要分配一张散列表，并用链接法解决碰撞，表中大约要存放n=2000个字符串，每个字符有8位')
        print(' 一次不成功的查找大约要检查3个元素，但我们并不在意，故分配散列表的大小为m=701.')
        print(' 之所以选择701这个数，是因为它是个接近a=2000/3，但又不接近2的任何幂次的质数。把每个关键字k视为一个整数')
        print(' 则有散列函数h(k) = k mod 701')
        print('11.3.2 乘法散列法')
        print(' 构造散列函数的乘法方法包含两个步骤。第一步，用关键字k乘上常数A(0<A<1),', 
            '并抽出kA的小数部分。然后，用m乘以这个值，再取结果的底(floor)。总之，散列函数为h(k)=[m(kA mod 1)]')
        print(' 其中kA mod 1 即kA的小数部分，亦即kA-[kA]')
        print(' 乘法方法的一个优点是对m的选择没有特别的要求，一般选择它为2的某个幂次m=2^p')
        print(' 虽然这个方法对任何的A值都适用，但对某些值效果更好。Knuth认为最佳的选择与待散列的数据的特征有关A=(sqrt(5)-1)/2=0.6180339887...就是一个比较理想的值')
        print(' 例子：假设有k=123456,p=14,m=2^14=16384,w=32,根据Knuth的建议')
        print(' 取A为形如s/2^32的分数，它与(sqrt(5)-1)/2最为接近，于是A=2654435769/2^32')
        print(' k*s=32770622297664=(76300*2^32)+17612864,从而有r1=76300,r0=17612864,r0的14个最高有效位产生了散列值h(k)=67')
        print('11.3.3 全域散列')
        print('如果让某个与你作对的人来选择要散列的关键字，那么他会选择全部散列到同一槽中的n个关键字，使得平均检索值为Θ(n)')
        print('任何一个特定的散列函数都可能出现这种最坏情况性态：唯一有效的改进方法是随机地选择散列函数，使之独立要存储的关键字。')
        print('这种方法称作全域散列(universal hashing),不管对手选择了怎样的关键字，其平均性态都很好')
        print('全域散列的基本思想是在执行开始时，就从一族仔细设计的函数中，随机地选择一个座位散列函数')
        print('就像在快速排序中一样，随机化保证了没有哪一种输入会导致最坏情况性态。')
        print('同时，随机化使得即使对同一个输入，算法在每一次执行时的性态也都不一样')
        print('这样就可以确保对于任何输入，算法都具有较好的平均情况性态')
        print('设H为有限的一组散列函数，它将给定的关键字域U映射到{0,1,..,m-1}中。这样的一个函数组称为是全域的')
        print('定理11.3 如果h选自一组全域的散列函数，并用于将n个关键字散列到一个大小为m的、用链接法解决碰撞的表T中')
        print('对于每一个关键字k，定义一个随机变量Yk,它等于非k的、与k散列到同一槽位中的其他关键字的数目')
        print('推论11.4 对于一个具有m个槽位的表，利用全域散列和链接法解决碰撞，需要Θ(n)的期望时间来处理任何包含了n')
        print(' 个INSERT,SEARCH和DELETE操作的操作序列，该序列中包含了O(m)个INSERT操作')
        print('证明：由于插入操作的数目为O(m),有n=O(m),从而a=O(1)。')
        print('INSERT操作和DELETE操作需要常量时间，根据定理11.3，每一个INSERT操作的期望时间为O(1)')
        print('于是，根据期望值的线性性质，整个操作序列的期望时间为O(n)')
        print('很容易设计出一个全域散列函数类，这一点只需一点点数论方面的知识即可加以证明')
        print('首先选择一个足够大的质数p，使得每一个可能的关键字k都落在0到p-1的范围内')
        print('由于p是一个质数，解决模p的方程。假定了关键字域的大小大于散列表中的槽位数，故有p>m')
        print('定义散列函数h，利用一次线性变换，后跟模p、再模m的归纳，有h=((ak+b) mod p) mod m')
        print('定理11.5 由上述公式定义的散列函数类是全域的')
        print('练习11.3-1 假设希望查找一个长度为n的链表，其中每一个元素都包含一个关键字k和一个散列值h。每一个关键字都是长字符串')
        print('练习11.3-2 假设一个长度为r的字符串被散列到m个槽中，方法是将其视为一个以128为基数的数，然后应用除法方法')
        print('练习11.3-3 考虑除法方法的另一种版本，其中h(k)=k mod n,m=2^p-1,k为按基数2^p解释的字符串')
        print(' 证明：如果串x可由串y通过其自身的置换排列导出，则x和y具有相同的散列')
        print('练习11.3-4 考虑一个大小为m=1000的散列表和对应一个散列函数h(k)=m(kA mod 1)')
        print(' A=(sqrt(5)-1)/5,计算61，62，63，64，65被映射到的位置')
        print('练习11.3-5 定义一个从有限集合U到有限集合B上的散列函数簇H为全域的')
        print('练习11.3-6 略')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_4:
    '''
    chpater11.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.4 note

        Example
        ====
        ```python
        Chapter11_4().note()
        ```
        '''
        print('chapter11.4 note as follow')
        print('11.4 开放寻址法')
        print('在开放寻址法中，所有的元素都存放在散列表里面，即每个表项或包含一个动态元素的集合，或包含nil')
        print('当检查一个元素时，要检查所有的表项，直到找到所有的元素，或者最终发现该元素不在表中')
        print('不像在链接法中，没有链表，也没有元素存放在散列表之外。')
        print('这种方法中，散列表可能存满，以至于不能插入新的元素。但是装载因子a是不可能超过1的')
        print('当然，也可以将用作链接的链表存放在散列表未用的槽中。但开放寻址法的好处是它根本不需要指针。')
        print('不用存储指针而节省空间，从而可以用同样的空间提供更多的槽，减小碰撞，提高查找速度')
        print('在开放寻址法中，当要插入一个元素时，可以连续地检查散列表的各项')
        print('直到找到一个空槽来存放插入的关键字为止。检查的顺序不一定是0,1,2...,m-1(这种顺序下查找时间为Θ(n))')
        print('开放寻址法中，对散列表元素的删除操作执行起来会比较困难')
        print('当从槽i中删除关键字时，不能仅将NIL置于其中标识它为空')
        print('有三种技术常用来计算开放寻址法中探查序列，线性探查，二次探查以及双重探查。')
        print('但是，这三种技术都不能实现一致散列的假设。')
        print('在这三种技术中，双重散列能产生的探查序列最多，因而能给出最好的结果')
        print('线性探查存在着一次群集问题，随着时间的推移，连续被占用的槽不断增加，平均查找的时间也不断增加')
        print('定理11.6 给定一个装载因子为a=n/m<1的开放寻址散列表，在一次不成功的查找中，期望的探查次数至多为1/(1-a),假设散列是一致的')
        print('推论11.7 平均情况下，向一个装载因子为a的开放寻址散列表插入一个元素时，至多只需要做1/(1-a)次探查，假设采用的是一次散列')
        print('定理11.8 给定一个装载因子为a<1的开放寻址散列表，一次成功查找中的期望探查数至多为1/aln1/(1-a)')
        print('假定散列是一致的，且表中的每个关键字被查找的可能性是相同的')
        print('练习11.4-1 考虑将关键字10、22、31、4、15、28、17、88、59用开放寻址法插入到一个长度为m=11的散列表中')
        print(' 主散列函数为h(k)=k mod m,说明用线性探查、二次探查以及双重散列h2(k)=1+(k mod (m-1))将这些关键字插入散列表的结果')
        print('练习11.4-2 请写出HASH-DELETE的伪代码；修改HASH-INSERT,使之能处理特殊值DELETED。')
        print('练习11.4-3 假设采用双重散列来解决碰撞；亦即，所用的散列函数为h(k,i)=(h1(k)+ih2(k)) mod m')
        print(' 证明如果对某个关键字k，m和h2(k)有最大公约数d>=1,则在对关键字k的一次不成功的查找中，在回到槽h1(k)之前，要检查散列表的1/d。')
        print(' 于是，当d=1时，m与h2(k)互质，查找操作可能要检查整个散列表。')
        print('练习11.4-4 考虑一个采用了一致散列的开放寻址散列表。给出当装载因子为3/4和7/8时')
        print(' 在一次不成功查找中期望探查数的上界，以及一次成功查找中期望探查数的上界')
        print('练习11.4-5 考虑一个装载因子为a的开放寻址散列表。给出一个非0值a，使得在一次不成功的查找中')
        print(' 期望的探查数等于成功查找中期望探查数的两倍。此处的两个期望探查数上界可以根据定理11.6和定理11.8得出')
        print('')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_5:
    '''
    chpater11.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.5 note

        Example
        ====
        ```python
        Chapter11_5().note()
        ```
        '''
        print('chapter11.5 note as follow')
        print('11.5 完全散列')
        print('人们之所以使用散列技术，主要是因为它有出色的性能，其实，当关键字集合是静态的时，散列技术还可以用来获得出色的最坏情况性能')
        print('所谓静态就是指一旦各关键字存入表中后，关键字集合就不再变化了')
        print('如果某一种散列技术在进行查找时，其最坏情况内存访问次数为O(1)的话，则称其为完全散列(perfect hashing)')
        print('设计完全散列方案的基本想法是比较简单的。利用一种两级的散列方案，每一级上都采用全域散列')
        print('第一级与带链接的散列基本上是一样的：利用从某一全域散列函数簇中仔细选出的一个散列函数h，将n个关键字散列到m个槽中')
        print('然而，我们不是对散列到槽j中的所有关键字建立一个链表，而是采用了一个较小的二次散列表Sj，与其相关的散列函数为hj')
        print('通过仔细地选取散列函数hj,可以确保在第二级上不出现碰撞')
        print('但是，为了能真正确保在第二级上不出现碰撞，需要让散列表Sj的大小mj为散列到槽j中的关键字数nj的平方')
        print('mj对nj的这种二次依赖关系看上去可能使得总体存储需求很大')
        print('后面会说明，通过适当地选择第一次散列函数，预期使用的总存储空间仍然为O(n)')
        print('定理11.9 如果利用从一个全域散列函数类中随机选出的散列函数h，将n个关键字存储在一个大小为m=n^2的散列表中，那么出现碰撞的概率小于1/2')
        print('证明：共有(n 2)对关键字可能发生碰撞，如果h是从一个全域散列函数类H中随机选出的话，每一对关键字碰撞的概率为1/m。')
        print('设X为一个随机变量，它统计了碰撞的次数，当m=n^2时，期望的碰撞次数为E[X]<1/2')
        print('定理11.10 如果利用从某一全域散列函数类中随机选出的散列函数h，来将n个关键字存储到一个大小为m=n的散列表中')
        print('推论11.11 如果利用从某一全域散列函数类中随机选出的散列函数h，来将n个关键字存储到一个大小为m=n的散列表中，并将每个二次散列表的大小置为m=n^2')
        print('  则在一个完全散列方案中，存储所有二次散列表所需的存储总量的期望值小于2n')
        print('推论11.12 如果利用从某一全域散列函数类中随机选出的散列函数h，来将n个关键字存储到一个大小为m=n的散列表中，并将每个二次散列表的大小置为m=n^2')
        print('  则用于存储所有二次散列表的存储总量超过4n的概率小于1/2')
        print('练习11.5-1 假设要将n个关键字插入到一个大小为m,采用了开放寻址法和一致散列技术的散列表中。')
        print('  设p(n,m)为没有碰撞发生的概率。证明：p(n,m)<=e^(-n(n-1)/2m)')
        print('  论证当n超过sqrt(m)时，不发生碰撞的概率迅速趋于0')
        print('思考题11-1 最长探查的界：用一个大小为m的散列表来存储n个数据项目，并且有n<=m/2。采用开放寻址法来解决碰撞问题')
        print(' a) 假设采用了一致散列，证明对于i=1,2,...,n,第i次插入需要严格多余k次探查的概率至多为2^-k')
        print( 'b) 证明：对于i=1,2,...,n, 第i次插入需要多于2lgn次探查的概率至多是1/n^2')
        print('  设随机变量Xi表示第i次插入所需要的探查数。在上面b)已证明Pr{Xi>2lgn}<=1/n^2')
        print('  设随机变量X=maxXi表示插入中所需探查数的最大值')
        print( 'c) 证明：Pr{X>2lgn}<=1/n')
        print( 'd) 证明:最长探查序列的期望长度为E[x]=O(lgn)')
        print('思考题11-2 链接法中槽大小的界')
        print(' 假设有一个含有n个槽的散列表，并用链接法来解决碰撞问题。另假设向表中插入n个关键字。')
        print(' 每个关键字被等可能地散列到每个槽中。设在所有关键字被插入后，M是各槽中所含关键字数的最大值')
        print('思考题11-3 二次探查')
        print(' 假设要在一个散列表(表中的各个位置为0,1,...,m-1)中查找关键字k，并假设有一个散列函数h将关键字空间映射到集合{0,1,...,m-1}，查找方法如下')
        print(' 1) 计算值i<-h(k),置j<-0')
        print(' 2) 探查位置i，如果找到了所需的关键字k，或如果这个位置是空的，则结束查找')
        print(' 3) 置j<-(j+1) mod m,i<-(i+j) mod m，则返回步骤2)')
        print('  设m是2的幂：')
        print('  a) 通过给出c1和c2的适当值，来证明这个方案是一般的\"二次探查\"法的一个实例')
        print('  b) 证明：在最坏情况下，这个算法要检查表中的每一个位置')
        print('思考题11-4 k全域散列和认证')
        print('  设H={h}为一个散列函数类，其中每个h将关键字域U映射到{0,1,...,m-1}上。称H是k全域的')
        print(' 如果对每个由k个不同的关键字<x(1),x(2),...,x(k)>构成的固定序列，以及从H中随机选出的任意h')
        print(' a) 证明：如果H是2全域的，则它是全域的。')
        print(' b) 设U为取自Zp中数值的n元组集合，并设B=Zp，此处p为质数')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

chapter11_1 = Chapter11_1()
chapter11_2 = Chapter11_2()
chapter11_3 = Chapter11_3()
chapter11_4 = Chapter11_4()
chapter11_5 = Chapter11_5()

def printchapter11note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter eleven!')  
    chapter11_1.note()
    chapter11_2.note()
    chapter11_3.note()
    chapter11_4.note()
    chapter11_5.note()

# python src/chapter11/chapter11note.py
# python3 src/chapter11/chapter11note.py
if __name__ == '__main__':  
    printchapter11note()
else:
    pass

```

```py

# python src/chapter12/chapter12note.py
# python3 src/chapter12/chapter12note.py
'''
Class Chapter12_1

Class Chapter12_2

Class Chapter12_3

Class Chapter12_4

'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

if __name__ == '__main__':
    from searchtree import SearchTree, SearchTreeNode, RandomSearchTree
else:
    from .searchtree import SearchTree, SearchTreeNode, RandomSearchTree

class Chapter12_1:
    '''
    chpater12.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.1 note

        Example
        ====
        ```python
        Chapter12_1().note()
        ```
        '''
        print('chapter12.1 note as follow')
        print('查找树(search tree)是一种数据结构，它支持多种动态集合操作，', 
            '包括SEARCH,MINIMUM,MAXIMUM,PREDECESSOR,SUCCESSOR,INSERT以及DELETE,', 
            '它既可以用作字典，也可以用作优先队列')
        print('在二叉查找树(binary search tree)上执行的基本操作时间与树的高度成正比')
        print('对于一颗含n个结点的完全二叉树，这些操作的最坏情况运行时间为Θ(lgn)')
        print('但是，如果树是含n个结点的线性链，则这些操作的最坏情况运行时间为Θ(n)')
        print('在12.4节中可以看到，一棵随机构造的二叉查找树的期望高度为O(lgn)，', 
            '从而这种树上基本动态集合操作的平均时间为Θ(lgn)')
        print('在实际中，并不总能保证二叉查找树是随机构造成的，但对于有些二叉查找树的变形来说，')
        print(' 各基本操作的最坏情况性能却能保证是很好的')
        print('第13章中给出这样一种变形，即红黑树，其高度为O(lgn)。第18章介绍B树，这种结构对维护随机访问的二级(磁盘)存储器上的数据库特别有效')
        print('12.1 二叉查找树')
        print('一颗二叉查找树是按二叉树结构来组织的。这样的树可以用链表结构表示，其中每一个结点都是一个对象。')
        print('结点中除了key域和卫星数据外，还包含域left,right和p，它们分别指向结点的左儿子、右儿子和父节点。')
        print('如果某个儿子结点或父节点不存在，则相应域中的值即为NIL，根结点是树中唯一的父结点域为NIL的结点')
        print('二叉查找树，对任何结点x，其左子树中的关键字最大不超过key[x],其右子树中的关键字最小不小于key[x]')
        print('不同的二叉查找树可以表示同一组值，在有关查找树的操作中，大部分操作的最坏情况运行时间与树的高度是成正比的')
        print('二叉查找树中关键字的存储方式总是满足以下的二叉树查找树性质')
        print('设x为二叉查找树中的一个结点，如果y是x的左子树的一个结点，则key[y]<=key[x].')
        print(' 如果y是x的右子树的一个结点，则key[x]<=key[y]')
        print('即二叉查找树的的某结点的左儿子总小于等于自身，右儿子总大于等于自身')
        print('根据二叉查找树的性质，可以用一个递归算法按排列顺序输出树中的所有关键字。')
        print('这种算法成为中序遍历算法，因为一子树根的关键字在输出时介于左子树和右子树的关键字之间')
        print('前序遍历中根的关键字在其左右子树中的关键字输出之前输出，', 
            '而后序遍历中根的关键字再其左右子树中的关键字之后输出')
        print('只要调用INORDER-TREE-WALK(root[T]),就可以输出一棵二叉查找树T中的全部元素')
        print('INORDER-TREE-WALK(x)')
        print('if x != None')
        print('  INORDER-TREE-WALK(left[x])')
        print('  print key[x]')
        print('  INORDER-TREE-WALK(right[x])')
        print('  遍历一棵含有n个结点的二叉树所需的时间为Θ(n),因为在第一次调用遍历过程后，', 
            '对树中的每个结点，该过程都要被递归调用两次')
        print('定理12.1 如果x是一棵包含n个结点的子树的根上，调用INORDER-TREE-WALK过程所需的时间。对于一棵空子树')
        print(' INORDER-TREE-WALK只需很少的一段常量时间(测试x!=None),因而有T(0)=c,c为某一正常数')
        print('练习12.1-1 基于关键字集合{1,4,5,10,16,17,21}画出高度为2,3,4,5,6的二叉查找树')
        print('练习12.1-2 二叉查找树性质与最小堆性质之间有什么区别。能否利用最小堆性质在O(n)时间内，')
        print('  按序输出含有n个结点的树中的所有关键字')
        print('练习12.1-3 给出一个非递归的中序树遍历算法')
        print('  有两种方法，在较为容易的方法中，可以采用栈作为辅助数据结构，在较为复杂的方法中，不采用栈结构')
        print('练习12.1-4 对一棵含有n个结点的树，给出能在Θ(n)时间内，完成前序遍历和后序遍历的递归算法')
        print('练习12.1-5 在比较模型中，最坏情况下排序n个元素的时间为Ω(nlgn),则为从任意的n个元素中构造出一棵二叉查找树')
        print('  任何一个基于比较的算法在最坏情况下，都要花Ω(nlgn)的时间')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_2:
    '''
    chpater12.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.2 note

        Example
        ====
        ```python
        Chapter12_2().note()
        ```
        '''
        print('chapter12.2 note as follow')
        print('12.2 查询二叉查找树')
        print('对于二叉查找树，最常见的操作是查找树中的某个关键字。')
        print('除了SEARCH操作外，二叉查找树还能支持诸如MINIMUM,MAXIMUM,SUCCESSOR和PREDECESSOR等查询')
        print('并说明对高度为h的树，它们都可以在O(h)时间内完成')
        print('查找')
        print(' 我们用下面的过程在树中查找一个给定的关键字。给定指向树根的指针和关键字k，', 
            '过程TREE-SEARCH返回包含关键字k的结点(如果存在的话)的指针，否则返回None')
        print('TREE-SEARCH(x,k)')
        print('if x = None or k=key[x]')
        print('  return x')
        print('if k < key[x]')
        print('  return TREE-SEARCH(left[x],k)')
        print('else')
        print('  return TREE-SEARCH(right[x],k)')
        print('最大关键字元素和最小关键字元素')
        print('要查找二叉树中具有最小关键字的元素，只要从根结点开始，沿着各结点的left指针查找下去，直至遇到None时为止')
        print('二叉查找树性质保证了TREE-MINIMUM的正确性。如果一个结点x无子树，', 
            '其右子树中的每个关键字都至少和key[x]一样大')
        print('对高度为h的树，这两个过程的运行时间都是O(h),这是因为，如在TREE-SEARCH过程中一样，', 
            '所遇到的结点序列构成了一条沿着结点向下的路径')
        print('前趋和后继')
        print('给定一个二叉查找树中的结点，有时候要求找出在中序遍历顺序下它的后继')
        print('如果所有的关键字均不相同，则某一结点x的后继即具有大于key[x]中关键字中最小者的那个结点')
        print('根据二叉查找树的结构，不用对关键字做任何比较，就可以找到某个结点的后继')
        print('定理12.2 对一棵高度为h的二叉查找树，动态集合操作SEARCH,MINIMUM,', 
            'MAXIMUM,SUCCESSOR和PREDECESSOR等的运行时间为O(h)')
        print('练习12.2-1 假设在某二叉查找树中，有1到1000之间的一些数，现要找出363这个数。')
        print(' 下列的结点序列中，哪一个不可能是所检查的序列 b),c),e)')
        print('a) 2,252,401,398,330,344,397,363')
        print('b) 924,220,911,244,898,258,362,363')
        print('c) 925,202,911,240,912,245,363')
        print('d) 2,399,387,219,266,382,381,278,363')
        print('e) 935,278,347,621,299,392,358,363')
        print('练习12.2-2 写出TREE-MINIMUM和TREE-MAXIMUM过程的递归版本')
        print('练习12.2-3 写出TREE-PREDECESSOR过程')
        print('练习12.2-4 假设在二叉查找树中，对某关键字k的查找在一个叶结点处结束，考虑三个集合')
        print(' A 包含查找路径左边的关键字；')
        print(' B 包含查找路径上的关键字；')
        print(' C 包含查找路径右边的关键字；')
        print(' 任何三个关键字a∈A,b∈B,c∈C 必定满足a<=b<=c,请给出该命题的一个最小可能的反例')
        print('练习12.2-5 性质：如果二叉查找树中的某结点有两个子女，则其后继没有左子女，其前趋没有右子女')
        print('练习12.2-6 考虑一棵其关键字各不相同的二叉查找树T。证明：如果T中某个结点x的右子树为空，且x有一个后继y')
        print('  那么y就是x的最低祖先，且其左孩子也是x的祖先。')
        print('练习12.2-7 对于一棵包含n个结点的二叉查找树，其中序遍历可以这样来实现；先用TREE-MINIMUM找出树中的最小元素')
        print(' 然后再调用n-1次TREE-SUCCESSOR。证明这个算法的运行时间为Θ(n)')
        print('练习12.2-8 证明：在一棵高度为h的二叉查找树中，无论从哪一个结点开始')
        print(' 连续k次调用TREE-SUCCESSOR所需的时间都是O(k+h)')
        print('练习12.2-9 设T为一棵其关键字均不相同的二叉查找树，并设x为一个叶子结点，y为其父结点。')
        print(' 证明：key[y]或者是T中大于key[x]的最小关键字，或者是T中小于key[x]的最大关键字')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_3:
    '''
    chpater12.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.3 note

        Example
        ====
        ```python
        Chapter12_3().note()
        ```
        '''
        print('chapter12.3 note as follow')
        print('12.3 插入和删除')
        print('插入和删除操作会引起二叉查找树表示的动态集合的变化，要反映出这种变化，就要修改数据结构')
        print('但在修改的同时，还要保持二叉查找树性质')
        print('插入一个新元素而修改树的结构相对来说比较简单,但在执行删除操作时情况要复杂一些')
        print('插入：为将一个新值v插入到二叉查找树T中，可以调用TREE-INSERT')
        print(' 传给该过程的参数是个结点z，并且有key[z]=v,left[z]=None,right[z]=None')
        print(' 该过程修改T和z的某些域，并把z插入到树中的合适位置')
        print('定理12.3 对高度为h的二叉查找树，动态集合操作INSERT和DELETE的运行时间为O(h)')
        tree = SearchTree()
        tree.insert_recursive(SearchTreeNode(12, 0))
        tree.insert(SearchTreeNode(11, 1))
        tree.insert(SearchTreeNode(10, 2))
        tree.insert(SearchTreeNode(15, 3))
        tree.insert_recursive(SearchTreeNode(9, 4))   
        print(tree.all())
        print(tree.count())
        print(tree.inorder_tree_walk(tree.root))
        print(tree.tree_search(tree.root, 15))
        print(tree.tree_search(tree.root, 8))
        print(tree.iterative_tree_search(tree.root, 10))
        print(tree.iterative_tree_search(tree.root, 7))
        print(tree.maximum(tree.root))
        print(tree.maximum_recursive(tree.root))
        print(tree.minimum(tree.root))
        print(tree.minimum_recursive(tree.root))
        print(tree.successor(tree.root))
        print(tree.predecessor(tree.root))
        print('练习12.3-1 TREE-INSERT的递归版本测试成功！')
        print('练习12.3-2 假设通过反复插入不同的关键字的做法来构造一棵二叉查找树。论证：为在树中查找一个关键字')
        print(' 所检查的结点数等于插入该关键字所检查的结点数加1')
        print('练习12.3-3 这个排序算法的最好时间和最坏时间:O(h) * n * O(h)', tree.allkey())
        print('练习12.3-4 假设另有一种数据结构中包含指向二叉查找树中某结点y的指针，并假设用过程TREE-DELETE来删除y的前趋z')
        print(' 这样做会出现哪些问题呢，如何改写TREE-DELETE来解决这些问题')
        print('练习12.3-5 删除操作是不可以交换的，先删除x再删除y和先删除y再删除x是不一样的，会影响树的结构')
        print('练习12.3-6 当TREE-DELETE中的结点z有两个子结点时，可以将其前趋(而不是后继)拼接掉')
        print(' 提出了一种公平的策略，即为前趋和后继结点赋予相同的优先级，从而可以得到更好地经验性能。')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

class Chapter12_4:
    '''
    chpater12.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter12.4 note

        Example
        ====
        ```python
        Chapter12_4().note()
        ```
        '''
        print('chapter12.4 note as follow')
        print('12.4 随机构造的二叉查找树')
        print('二叉查找树上各基本操作的运行时间都是O(h),h为树的高度。', 
            '但是，随着元素的插入或删除，树的高度会发生变化')
        print('例如各元素是按严格增长的顺序插入的，那么构造出来的树就是一个高度为n-1的链')
        print('要使树的高度尽量平均最小，所以采用随机化技术来随机插入结点')
        print('插入顺序不同则树的结构不同')
        print('如在快速排序中那样，可以证明其平均情况下的行为更接近于最佳情况下的行为，', 
            '而不是接近最坏情况下的行为')
        print('不幸的是，如果在构造二叉查找树时，既用到了插入操作，又用到了删除，那么就很难确定树的平均高度到底是多少')
        print('如果仅用插入操作来构造树，则分析相对容易些。可以定义在n个不同的关键字上的一棵随机构造的二叉查找树')
        print('它是通过按随机的顺序，将各关键字插入一棵初始为空的树而形成的，并且各输入关键字的n!种排列是等可能的')
        print('这一概念不同于假定n个关键字上的每棵二叉查找树都是等可能的')
        print('这一节要证明对于在n个关键字上随机构造的二叉查找树，其期望高度为O(lgn)。假设所有关键字都是不同的')
        print('首先定义三个随机变量，它们有助于测度一棵随机构造的二叉查找树的高度：Xn表示高度')
        print('定义指数高度Yn=2^Xn,Rn表示一个随机变量，存放了该关键字在这n个关键字中的序号')
        print('Rn的值取集合{1,2,...,n}中的任何元素的可能性都是相同的')
        print('定理12.4：一棵在n个关键字上随机构造的二叉查找树的期望高度为O(lgn)')
        random_tree = RandomSearchTree()
        random_tree.randominsertkey(1)
        random_tree.randominsertkey(2)
        random_tree.randominsertkey(3)
        random_tree.randominsertkey(3)
        random_tree.randominsertkey(4)
        random_tree.randominsertkey(5)
        random_tree.update()
        random_tree.insertkey(0)
        print(random_tree.all())
        print(random_tree.allkey())
        print(random_tree.inorder_tree_walk(random_tree.root))
        print('练习12.4-1 证明恒等式∑i=0,n-1(i+3, 3)=(n+3, 4)')
        print('练习12.4-2 请描述这样一个的一棵二叉查找树：其中每个结点的平均深度Θ(lgn)，但是树的深度为ω(lgn)')
        print(' 对于一棵含n个结点的二叉查找树，如果其中每个结点的平均深度为Θ(lgn),给出其高度的一个渐进上界O(nlgn)')
        print('练习12.4-3 说明基于n个关键字的随机选择二叉查找树概念(每棵包含n个结点的树被选到的可能性相同)，与本节中介绍的随机构造二叉查找树的概念是不同的')
        print('练习12.4-4 证明f(x)=2^x是凸函数')
        print('练习12.4-5 现对n个输入数调用RANDOMIZED-QUICKSORT。', 
            '证明：对任何常数k>0,输入数的所有n!中排列中，除了其中的O(1/n^k)中排列之外，都有O(nlgn)的运行时间')
        print('思考题12-1 具有相同关键字的二叉查找树')
        print(' 具有相同关键字的存在，给二叉查找树的实现带来了一些问题')
        print(' 当用TREE-INSERT将n个具有相同关键字的数据项插入到一棵初始为空的二叉查找树中，该算法的渐进性能如何')
        print(' 可以对TREE-INSERT做一些改进，即在第5行的前面测试key[z]==key[x],在第11行前面测试key[z]==key[y]')
        print(' 在结点x处设一个布尔标志b[x],并根据b[x]的不同值，置x为left[x]或right[x],', 
            '每当插入一个与x具有相同关键字的结点时，b[x]取TRUE或FALSE,随机地将x置为left[x]或right[x]')
        print('思考题12-2 基数树RadixTree数据结构')
        print(' 给定两个串a=a0a1...ap和b=b0b1...bp,其中每一个ai和每一个bj都属于某有序字符集')
        print(' 例如，如果a和b是位串,则根据规则10100<10110, 10100<101000,这与英语字典中的排序很相似')
        print(' 设S为一组不同的二进制串构成的集合，各串的长度之和为n，说明如何咯用基数树，在Θ(n)时间内将S按字典序排序')
        print('思考题12-3 随机构造的二叉查找树中的平均结点深度')
        print('证明在一棵随机构造的二叉查找树中，n个结点的平均深度为O(lgn)')
        print(' 与RANDOMIZED-QUICKSORT的运行机制之间的令人惊奇的相似性')
        print(' 定义一棵二叉树T的总路径长度P(T)为T中所有结点x的深度之和，表示d(x,T)')
        print(' T中结点的平均深度为P(T)/n,进而证明P(T)的期望值为O(nlgn)')
        print(' 设TL和TR分别表示树T的左右子树，论证：如果T有n个结点，则有：P(T)=P(TL)+P(TR)+n-1')
        print(' 设P(n)表示一棵包含n个结点的随机构造二叉树中的平均总路径长度')
        print('在对快速排序算法的每一次递归调用中，都是随机地选择一个支点元素来对待排序元素集合进行划分')
        print(' 二叉查找树中的每个结点也对以该结点为根的子树中的所有元素进行划分')
        print('思考题12-4 不同的二叉树数目')
        print(' 设bn表示包含n个结点的不同的二叉树的数目，在本问题里，要给出关于bn的公式和一个渐进估计')
        # python src/chapter12/chapter12note.py
        # python3 src/chapter12/chapter12note.py

chapter12_1 = Chapter12_1()
chapter12_2 = Chapter12_2()
chapter12_3 = Chapter12_3()
chapter12_4 = Chapter12_4()

def printchapter12note():
    '''
    print chapter12 note.
    '''
    print('Run main : single chapter twelve!')  
    chapter12_1.note()
    chapter12_2.note()
    chapter12_3.note()
    chapter12_4.note()

# python src/chapter12/chapter12note.py
# python3 src/chapter12/chapter12note.py
if __name__ == '__main__':  
    printchapter12note()
else:
    pass

```

```py

from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

import time as _time
from random import randint as _randint

class SearchTreeNode:
    '''
    二叉查找树的结点
    '''
    def __init__(self, key, index, \
        p = None, left = None, right = None):
        '''

        二叉树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + ','\
                'index:' + str(self.index)

class SearchTree:
    '''
    二叉查找树
    '''
    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []

    def inorder_tree_walk(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None:
            array.append(str(x))
            array = array + right
        return array

    def __inorder_tree_walk_key(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.__inorder_tree_walk_key(x.left)
            array = array + left
            right = self.__inorder_tree_walk_key(x.right)  
        if x != None:
            array.append(x.key)
            array = array + right
        return array

    def tree_search(self, x : SearchTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != None and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return None

    def iterative_tree_search(self, x : SearchTreeNode, key):
        '''
        查找的非递归版本

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x != None:
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x
        return x

    def minimum(self, x : SearchTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != None:
            x = x.left
        return x

    def __minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def maximum(self, x : SearchTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != None:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def successor(self, x : SearchTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.right != None:
            return self.minimum(x.right)
        y = x.p
        while y != None and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : SearchTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != None:
            return self.maximum(x.left)
        y = x.p
        while y != None and x == y.left:
            x = y
            y = y.p
        return y

    def insertkey(self, key, index = None):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        self.insert(SearchTreeNode(key, index))

    def insert(self, z : SearchTreeNode):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        y = None
        x = self.root
        while x != None:
            y = x
            if z.key < x.key:
                x = x.left
            elif z.key > x.key:
                x = x.right
            else:
                # 处理相同结点的方式，随机分配左右结点
                if _randint(0, 1) == 0:
                    x = x.left
                else:
                    x = x.right
        z.p = y
        if y == None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        elif z.key > y.key:
            y.right = z
        else:
            # 处理相同结点的方式，随机分配左右结点
            if _randint(0, 1) == 0:
                y.left = z
            else:
                y.right = z
        self.nodes.append(z) 
        self.lastnode = z

    def insertnodes(self, nodes : list):
        '''
        按顺序插入一堆结点
        '''
        for node in nodes:
            if node is type(SearchTreeNode):
                self.insert(node)
            else:
                self.insertkey(node)

    def __insertfrom(self, z : SearchTreeNode, x : SearchTreeNode, lastparent : SearchTreeNode):
        if x != None:
            if z.key < x.key:
                self.__insertfrom(z, x.left, x)
            else:
                self.__insertfrom(z, x.right, x)
        else:
            z.p = lastparent
            if z.key < lastparent.key:
                lastparent.left = z
            else:
                lastparent.right = z

    def insert_recursive(self, z : SearchTreeNode):
        '''
        插入元素(递归版本)，时间复杂度`O(h)` `h`为树的高度
        '''
        if self.root == None:
            self.root = z
        else:  
            self.__insertfrom(z, self.root, None)
        self.nodes.append(z) 
        self.lastnode = z

    def delete(self, z : SearchTreeNode):
        '''
        删除操作，时间复杂度`O(h)` `h`为树的高度
        '''
        if z.left == None or z.right == None:
            y = z
        else:
            y = self.successor(z)
        if y.left != None:
            x = y.left
        else:
            x = y.right
        if x != None:
            x.p = y.p
        if y.p == None:
            self.root = x
        else:
            if y == y.p.left:
                y.p.left = x
            else:
                y.p.right = x
        if y != None:
            z.key = y.key
            z.index = _deepcopy(y.index)
        self.nodes.remove(z) 
        return y
        
    def all(self):
        '''
        返回二叉查找树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def allkey(self):
        '''
        按升序的方式输出所有结点`key`值构成的集合
        '''
        return self.__inorder_tree_walk_key(self.root)

    def count(self):
        '''
        二叉查找树中的结点总数
        '''
        return len(self.nodes)

    def depth(self, x : SearchTreeNode):
        '''
        二叉查找树`x`结点的深度
        '''
        if x == None:
            return 0
        return 1 + max(self.depth(x.left), self.depth(x.right))

    def leftrotate(self, x : SearchTreeNode):
        '''
        左旋 时间复杂度:`O(1)`
        '''
        if x.right == None:
            return
        y : SearchTreeNode = x.right
        x.right = y.left
        if y.left != None:
            y.left.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y

    def rightrotate(self, x : SearchTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        if x.left == None:
            return
        y : SearchTreeNode = x.left
        x.left = y.right
        if y.right != None:
            y.right.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y

class RandomSearchTree(SearchTree):

    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []
        self.__buffers = []

    def __randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        ===
        `array` : 随机排列前的数组

        Return
        ===
        `random_array` : 随机排列后的数组

        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def randominsert(self, z : SearchTreeNode):
        '''
        使用随机化技术插入结点到缓存
        '''
        self.__buffers.append(z)

    def randominsertkey(self, key, index = None):
        '''
        使用随机化技术插入结点到缓存
        '''
        z = SearchTreeNode(key, index)
        self.randominsert(z)

    def update(self):
        '''
        从缓存更新二叉查找树结点
        '''
        randombuffers = self.__randomize_inplace(self.__buffers)
        for buffer in randombuffers:
            self.insert(buffer)
        self.__buffers.clear()
 
if __name__ == '__main__':
    tree = SearchTree()
    node1 = SearchTreeNode(12, 0)
    node2 = SearchTreeNode(11, 1)
    node3 = SearchTreeNode(10, 2)
    node4 = SearchTreeNode(15, 3)
    node5 = SearchTreeNode(9, 4)
    tree.insert_recursive(node1)
    tree.insert(node2)
    tree.insert(node3)
    tree.insert(node4)
    tree.insert_recursive(node5)   
    print(tree.all())
    print(tree.count())
    print(tree.inorder_tree_walk(tree.root))
    print(tree.tree_search(tree.root, 15))
    print(tree.tree_search(tree.root, 8))
    print(tree.iterative_tree_search(tree.root, 10))
    print(tree.iterative_tree_search(tree.root, 7))
    print(tree.maximum(tree.root))
    print(tree.maximum_recursive(tree.root))
    print(tree.minimum(tree.root))
    print(tree.minimum_recursive(tree.root))
    print(tree.successor(tree.root))
    print(tree.predecessor(tree.root))
    print(tree.depth(tree.root))
    tree.insertkey(18)
    tree.insertkey(16)
    tree.leftrotate(node4)
    tree.insertkey(20)
    tree.rightrotate(node3)
    tree.insertkey(3)
    print(tree.all())
    random_tree = RandomSearchTree()
    random_tree.randominsertkey(1)
    random_tree.randominsertkey(2)
    random_tree.randominsertkey(3)
    random_tree.randominsertkey(4)
    random_tree.randominsertkey(5)
    random_tree.update()
    random_tree.insertkey(0)
    print(random_tree.all())
    print(random_tree.allkey())
    print(random_tree.inorder_tree_walk(random_tree.root))
    # python src/chapter12/searchtree.py
    # python3 src/chapter12/searchtree.py
else:
    pass
```

```py

# python src/chapter13/chapter13note.py
# python3 src/chapter13/chapter13note.py
'''
Class Chapter13_1

Class Chapter13_2

Class Chapter13_3

Class Chapter13_4

'''

from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import sys as _sys
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange

if __name__ == '__main__':
    import redblacktree as rb
else:
    from . import redblacktree as rb

class Chapter13_1:
    '''
    chpater13.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.1 note

        Example
        ====
        ```python
        Chapter13_1().note()
        ```
        '''
        print('chapter13.1 note as follow')
        print('第13章 红黑树')
        print('由12章可以知道，一棵高度为h的查找树可以实现任何一种基本的动态集合操作')
        print('如SEARCH,PREDECESOR,MINIMUM,MAXIMUM,DELETE,INSERT,其时间都是O(h)')
        print('所以，当树的高度比较低时，以上操作执行的就特别快')
        print('当树的高度比较高时，二叉查找树的性能可能不如链表好，')
        print('红黑树是许多平衡的查找树中的一种，能保证在最坏情况下，基本的动态集合操作的时间为O(lgn)')
        print('13.1 红黑树的性质')
        print('红黑树是一种二叉查找树，但在每个结点上增加一个存储位表示结点的颜色，可以是RED，可以是BLACK')
        print('通过对任何一条从根到叶子的路径上各个结点的着色方式的限制，红黑树确保没有一条路径会比其他路径长出两倍，因而是接近平衡的')
        print('树中每个结点包含5个域，color,key,p,left,right')
        print('如果某结点没有子结点或者父结点，则该结点相应的域为NIL')
        print('将NONE看作指向二叉查找树的外结点，把带关键字的结点看作树的内结点')
        print('一颗二叉查找树满足下列性质，则为一颗红黑树')
        print('1.每个结点是红的或者黑的')
        print('2.根结点是黑的')
        print('3.每个外部叶结点(NIL)是黑的')
        print('4.如果一个结点是红的，则它的两个孩子都是黑的')
        print('5.对每个结点，从该结点到其子孙结点的所有路径上包含相同数目的黑结点')
        print('为了处理红黑树代码中的边界条件，采用一个哨兵来代替NIL，')
        print('对一棵红黑树T，哨兵NIL[T]是一个与树内普通结点具有相同域的对象，', 
            '它的color域为BLACK，其他域的值可以随便设置')
        print('通常将注意力放在红黑树的内部结点上，因为存储了关键字的值')
        print('在本章其余部分，画红黑树时都将忽略其叶子')
        print('从某个结点x出发，到达一个叶子结点的任意一条路径上，', 
            '黑色结点的个数称为该结点的黑高度，用bh(x)表示')
        print('引理13.1 一棵有n个内结点的红黑树的高度至多为2lg(n+1)')
        print(' 红黑树中某一结点x为根的子树中中至少包含2^bh(x)-1个内结点')
        print(' 设h为树的高度，从根到叶结点(不包括根)任意一条简单路径上', 
            '至少有一半的结点至少是黑色的；从而根的黑高度至少是h/2,也即lg(n+1)')
        print('红黑树是许多平衡的查找树中的一种，能保证在最坏情况下，', 
            '基本的动态集合操作SEARCH,PREDECESOR,MINIMUM,MAXIMUM,DELETE,INSERT的时间为O(lgn)')
        print('当给定一个红黑树时，第12章的算法TREE_INSERT和TREE_DELETE的运行时间为O(lgn)')
        print('这两个算法并不直接支持动态集合操作INSERT和DELETE，',
            '但并不能保证被这些操作修改过的二叉查找树是一颗红黑树')
        print('练习13.1-1 红黑树不看颜色只看键值的话也是一棵二叉查找树，只是比较平衡')
        print(' 关键字集合当中有15个元素，所以红黑树的最大黑高度为lg(n+1),n=15,即最大黑高度为4')
        print('练习13.1-2 插入关键字36后，36会成为35的右儿子结点，虽然红黑树中的每个结点的黑高度没有变')
        print(' 如果35是一个红结点，它的右儿子36却是红结点，违反了红黑树性质4，所以插入元素36后不是红黑树')
        print(' 如果35是一个黑结点，则关键字为30的结点直接不满足子路径黑高度相同，所以插入元素36后不是红黑树')
        print('练习13.1-3 定义松弛红黑树为满足性质1,3,4和5，不满足性质2(根结点是黑色的)')
        print(' 也就是说根结点可以是黑色的也可以是红色的，考虑一棵根是红色的松弛红黑树T')
        print(' 如果将T的根部标记为黑色而其他都不变，则所得到的是否还是一棵红黑树')
        print(' 是吧，因为跟结点的颜色改变不影响其子孙结点的黑高度，而根结点自身的颜色与自身的黑高度也没有关系')
        print('练习13.1-4 假设一棵红黑树的每一个红结点\"结点\"吸收到它的黑色父节点中，来让红结点的子女变成黑色父节点的子女')
        print(' 其可能的度是多少，次结果树的叶子深度怎样，因为红黑树中根的黑高度至少是h/2，红结点都被吸收的话，叶子结点变为原来的一半')
        print('练习13.1-5 红黑树一个定理：在一棵红黑树中，从某结点x到其后代叶子结点的所有简单路径中，最长的一条是最短一条的至多两倍')
        print('练习13.1-6 因为一棵有n个内结点的红黑树的高度至多为2lg(n+1),则高度为k，内结点个数最多为[2^(k/2)]-1')
        print('练习13.1-7 请描述出一棵在n个关键字上构造出来的红黑树，使其中红的内结点数与黑的内结点数的比值最大')
        print(' 这个比值是多少，具有最小可能比例的树又是怎样？此比值是多少')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_2:
    '''
    chpater13.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.2 note

        Example
        ====
        ```python
        Chapter13_2().note()
        ```
        '''
        print('chapter13.2 note as follow')
        print('13.2 旋转')
        print('当在含n个关键字的红黑树上运行时，查找树操作TREE-INSERT和TREE-DELETE的时间为O(lgn)')
        print('由于这两个操作对树作了修改，结果可能违反13.1节中给出的红黑性质。',
            '为保持这些性质，就要改变树中某些结点的颜色以及指针结构')
        print('指针结构的修改是通过旋转来完成的，这是一种能保持二叉查找树性质的查找树局部操作')
        print('给出左旋和右旋。当某个结点x上做左旋时，假设它的右孩子不是nil[T],',
            'x可以为树内任意右孩子不是nil[T]的结点')
        print('左旋以x到y之间的链为\"支轴\"进行，它使y成为该该子树新的根，x成为y的左孩子，而y的左孩子则成为x的右孩子')
        print('在LEFT-ROTATE的代码中，必须保证right[x]!=None,且根的父结点为None')
        print('练习13.2-1 RIGHT-ROTATE的代码已经给出')
        print('练习13.2-2 二查查找树性质：在一棵有n个结点的二叉查找树中，刚好有n-1种可能的旋转')
        print('练习13.2-3 属于x结点的子孙结点，当结点x左旋时，x的子孙结点的深度加1')
        print('练习13.2-4 二查查找树性质：任何一棵含有n个结点的二叉查找树，可以通过O(n)次旋转，',
            '转变为另外一棵含n个结点的二叉查找树')
        print('练习13.2-5 如果二叉查找树T1可以右转成二叉查找树T2，则可以调用O(n^2)次RIGHT-ROTATE来右转')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_3:
    '''
    chpater13.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.3 note

        Example
        ====
        ```python
        Chapter13_3().note()
        ```
        '''
        print('chapter13.3 note as follow')
        print('13.3 插入')
        print('一棵含有n个结点的红黑树中插入一个新结点的操作可以在O(lgn)时间内完成')
        print('红黑树T插入结点z时，就像是T是一棵普通的二叉查找树那样，然后将z染成红色，')
        print('为保证红黑性质能继续保持，调用一个辅助程序对结点重新旋转和染色，假设z的key域已经提前赋值')
        print('插入结点后，如果有红黑性质被破坏，则至多有一个被破坏，并且不是性质2就是性质4')
        print('如果违反性质2，则发生的原因是z的根而且是红的，')
        print('如果违反性质4，则原因是z和z的父结点都是红的')
        print('循环结束是因为z.p是黑的，')
        print('在while循环中需要考虑六种情况，其中三种与另外三种')
        print('情况1与情况2，3的区别在于z的父亲的兄弟(或叔叔)的颜色有所不同，')
        print('情况1:z的叔叔y是红色的')
        print('情况2:z的叔叔y是黑色的，而且z是右孩子')
        print('情况3:z的叔叔y是黑色的，而且z是左孩子')
        print('有趣的是，insert_fixup的整个过程旋转的次数从不超过两次')
        tree = rb.RedBlackTree()
        tree.insertkey(41)
        tree.insertkey(38)
        tree.insertkey(31)
        tree.insertkey(12)
        tree.insertkey(19)
        tree.insertkey(8)
        tree.insertkey(1)
        print('练习13.3-1 红黑树假设插入的结点x是红色的，但是将结点假设为黑色，则红黑树的性质4就不会破坏')
        print(' 但是不会这么做，原因是会直接改变其父亲结点的黑高度，破坏红黑树的性质5，这样会使红黑树的插入变得非常复杂')
        print('练习13.3-2 ', tree.inorder_tree_walk(tree.root))
        print('练习13.3-3 略')
        print('练习13.3-4 红黑树性质：RB-INSERT-FIXUP过程永远不会将color[nil[T]]设置为RED')
        print('练习13.3-5 考虑用RB-INSERT插入n个结点而成的一棵红黑树。证明：如果n>1,n=2时，则该树至少有一个红结点，',
            '第一个结点就是父节点一定为黑色结点，第二个插入结点则一定为红结点')
        print('练习13.3-6 如果红黑树的表示中不提供父指针的话，应当如何有效地实现RB-INSERT')
        print(' 不会')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

class Chapter13_4:
    '''
    chpater13.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter13.4 note

        Example
        ====
        ```python
        Chapter13_4().note()
        ```
        '''
        print('chapter13.4 note as follow')
        print('13.4 删除')
        print('和n个结点的红黑树上的其他基本操作一样，对一个结点的删除要花O(lgn)时间。', 
            '与插入操作相比，删除操作只是稍微复杂些')
        print('程序RB-DELETE是对TREE-DELETE程序(12.3)略作修改得来的。',
            '在删除一个结点后，该程序就调用一个辅助程序RB-DELETE-FIXUP,用来改变结点的颜色并作旋转，从而保持红黑树性质')
        print('过程TREE-DELETE和RB-DELETE之间有三点不同。首先，', 
            'TREE-DELETE中所有对NIL的引用在RB-DELETE中都被替换成哨兵Nil的引用')
        print('如果y是红色的，则当y被删除后，红黑性质仍然得以保持，理由如下：')
        print(' 树中各结点的黑高度都没有变化')
        print(' 不存在两个相邻的红色结点')
        print(' 因为如果y是红的，就不可能是根，所以根仍然是黑色的')
        print('传递给RB-DELETE-FIXUP的结点x是两个结点中的一个：在y被删除之前，如果y有个不是哨兵nil的孩子')
        print('则x为y的唯一孩子；如果y没有孩子，则x为哨兵nil,在后一种情况中，',
            '之后的无条件赋值保证了无论x是有关键字的内结点或哨兵nil,x现在的父结点都为先前y的父结点')
        print('在RB-DELETE中，如果被删除的结点y是黑色的，则会产生三个问题。')
        print(' 首先，如果y原来是根结点，而y的一个红色的孩子成为了新的根，就会违反性质2')
        print(' 其次，如果x和y.p都是红的，就会违反性质4')
        print(' 第三，删除y将导致先前包含y的任何路径上黑结点的个数少1，因此性质5被y的一个祖先破坏')
        print('  补救这个问题的一个办法就是把结点x视为还有额外的一重黑色。')
        print('RB-DELETE-FIXUP程序负责恢复性质1,2,4')
        print('while循环的目标是将额外的黑色沿树上移动,直到')
        print(' 1.x指向一个红黑结点，此时将x着色为黑色')
        print(' 2.x指向根，这是可以简单地消除那个额外的黑色，或者')
        print(' 3.做必要的旋转和颜色修改')
        print('RB-DELETE-FIXUP程序的几种情况')
        print(' 情况1.x的兄弟w是红色的')
        print(' 情况2.x的兄弟w是黑色的，而且w的两个孩子都是黑色的')
        print(' 情况3.x的兄弟w是黑色的，w的左孩子是红色的，右孩子是黑色的')
        print(' 情况4.x的兄弟w是黑色的，而且w的右孩子是红色的')
        print('RB-DELETE的运行时间：含n个结点的红黑树的高度为O(lgn),',
            '不调用RB-DELETE-FIXUP时该程序的总时间代价为O(lgn)')
        print('在RB-DELETE-FIXUP中，情况1,3和4在各执行一定次数的颜色修改和至多修改三次旋转后便结束')
        print('情况2是while循环唯一可以重复的情况，其中指针x沿树上升的次数至多为O(lgn)次，且不执行任何旋转')
        print('所以，过程RB-DELETE-FIXUP要花O(lgn)时间，做至多三次旋转，从而RB-DELETE的总时间为O(lgn)')
        print('练习13.4-1: 红黑树删除过程性质：在执行RB-DELETE-FIXUP之后，树根总是黑色的')
        print('练习13.4-2: 在RB-DELETE中，如果x和p[y]都是红色的，则性质4可以通过调用RB-DELETE-FIXUP(T,x)来恢复')
        print('练习13.4-3: 在练习13.3-2中，将关键字41,38,31,12,19,8连续插入一棵初始为空的树中，从而得到一棵红黑树。')
        print(' 请给出从该树中连续删除关键字8,12,19,31,38,41后的结果')
        tree = rb.RedBlackTree()
        nodekeys = [41, 38, 31, 12, 19, 8]
        for key in nodekeys:
            tree.insertkey(key)
        print(tree.all())
        nodekeys.reverse()
        for key in nodekeys:
            tree.deletekey(key)
        print(tree.all())
        tree.insertkey(1)
        print(tree.all())
        print('练习13.4-4: 在RB-DELETE-FIXUP的哪些行中，可能会检查或修改哨兵nil[T]?')
        print('练习13.4-5: ')
        print('练习13.4-6: x.p在情况1的开头一定是黑色的')
        print('练习13.4-7: 假设用RB-INSERT来将一个结点x插入一棵红黑树，紧接着又用RB-DELETE将它从树中删除')
        print(' 结果的红黑树与初始的红黑树是否相同？')
        print('思考题13-1: 持久动态集合')
        print(' 在算法的执行过程，会发现在更新一个动态集合时，需要维护其过去的版本，这样的集合被称为是持久的')
        print(' 实现持久集合的一种方法是每当该集合被修改时，就将其整个地复制下来，但是这种方法会降低一个程序的执行速度，而且占用过多的空间。')
        print(' 考虑一个有INSERT，DELETE和SEARCH操作的持久集合S，对集合的每一个版本都维护一个单独的根')
        print(' 为把关键字5插入的集合中去，就要创建一个具有关键字5的新结点,最终只是复制了树的一部分，新树和老树之间共享一些结点')
        print(' 假设树中每个结点都有域key,left,right,但是没有父结点的域')
        print(' 1.对一棵一般的持久二叉查找树，为插入一个关键字k或删除一个结点y，确定需要改变哪些结点')
        print(' 2.请写出一个程序PERSISTENT-TREE-INSERT,使得在给定一棵持久树T和一个要插入的关键字k时，它返回将k插入T后新的持久树T1')
        print(' 3.如果持久二叉查找树T的高度为h，所实现的PERSISTENT-TREE-INSERT的时间和空间要求分别是多少')
        print(' 4.假设我们在每个结点中增加一个父亲结点域。这样一来：PERSISTENT-TREE-INSERT需要做一些额外的复制工作')
        print('  证明在这种情况下。PERSISTENT-TREE-INSERT的时空要求Ω(n),其中n为树中的结点个数')
        print(' 5.说明如何利用红黑树来保证每次插入或删除的最坏情况运行时间为O(lgn)')
        print('思考题13-2: 红黑树上的连接操作')
        print(' 连接操作以两个动态集合S1和S2和一个元素x为参数，使对任何x1属于S1和x2属于S2')
        print(' 有key[x1]<=key[x]<=key[x2],该操作返回一个集合S=S1 ∪ {x} ∪ S2。')
        print(' 在这个问题中，讨论在红黑树上实现连接操作')
        print(' 1.给定一棵红黑树T，其黑高度被存放在域bh[T]。证明不需要树中结点的额外存储空间和', 
            '不增加渐进运行时间的前提下，可以用RB-INSERT和RB-DELETE来维护这个域')
        print(' 希望实现RB-JOIN(T1,x,T2),它删除T1和T2，并返回一棵红黑树T= T1 ∪ {x} ∪ T2')
        print(' 设n为T1和T2中的总结点数')
        print(' 证明RB-JOIN的运行时间是O(lgn)')
        print('思考题13-3: AVL树是一种高度平衡的二叉查找树:对每一个结点x，x的左子树与右子树的高度至多为1')
        print(' 要实现一棵AVL树，我们在每个结点内维护一个额外的域:h(x),即结点的高度。至于任何其他的二叉查找树T，假设root[T]指向根结点')
        print(' 1.证明一棵有n个结点的AVL树其高度为O(lgn)。证明在一个高度为h的AVL树中，至少有Fh个结点，其中Fh是h个斐波那契数')
        print(' 2.为把结点插入到一棵AVL树中，首先以二叉查找树的顺序把结点放在适当的位置上')
        print('  这棵树可能就不再是高度平衡了。具体地，某些结点的左子树与右子树的高度差可能会到2')
        print('  请描述一个程序BALANCE(x),输入一棵以x为根的子树，其左子树与右子树都是高度平衡的，而且它们的高度差至多是2')
        print('  即|h[right[x]]-h[left[x]]|<=2,然后将以x为根的子树转变为高度平衡的')
        print(' 3.请给出一个由n个结点的AVL树的例子，其中一个AVL-INSERT操作将执行Ω(lgn)次旋转')
        print('思考题13-4: Treap')
        print(' 如果将一个含n个元素的集合插入到一棵二叉查找树中，所得到的树可能会非常不平衡，从而导致查找时间很长')
        print(' 随机构造的二叉查找树往往是平衡的。因此，一般来说，要为一组固定的元素建立一棵平衡树，可以采用的一种策略')
        print(' 就是先随机排列这些元素，然后按照排列的顺序将它们插入到树中')
        print(' 如果一次收到一个元素，也可以用它们来随机建立一棵二叉查找树')
        print(' 一棵treap是一棵修改了结点顺序的二叉查找树。')
        print(' 通常树内的每个结点x都有一个关键字值key[x]。')
        print(' 另外，还要为结点分配priority[x],它是一个独立选取的随机数。假设所有的优先级都是不同的')
        print(' 在将一个结点插入一棵treap树内时，所执行的旋转期望次数小于2')
        print('AVL树：最早的平衡二叉树之一。应用相对其他数据结构比较少，windows对进程地址空间的管理用到了AVL树,平衡度也最好')
        print('红黑树：平衡二叉树，广泛应用在C++的STL中。如map和set都是用红黑树实现的')
        print('B/B+树：用在磁盘文件组织，数据索引和数据库索引')
        print('Trie树(字典树)：用在统计和排序大量字符串，如自动机')
        # python src/chapter13/chapter13note.py
        # python3 src/chapter13/chapter13note.py

chapter13_1 = Chapter13_1()
chapter13_2 = Chapter13_2()
chapter13_3 = Chapter13_3()
chapter13_4 = Chapter13_4()

def printchapter13note():
    '''
    print chapter13 note.
    '''
    print('Run main : single chapter thirteen!')  
    chapter13_1.note()
    chapter13_2.note()
    chapter13_3.note()
    chapter13_4.note()

# python src/chapter13/chapter13note.py
# python3 src/chapter13/chapter13note.py
if __name__ == '__main__':  
    printchapter13note()
else:
    pass

```

```py

from __future__ import division, absolute_import, print_function
from copy import deepcopy as _deepcopy

BLACK = 0
RED = 1

class RedBlackTreeNode:
    '''
    红黑树结点
    '''
    def __init__(self, key, index = None, color = RED, \
        p = None, left = None, right = None):
        '''
        红黑树树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.color = color
        self.p = p

    def __str__(self):
        '''
        str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})
        '''
        if self.isnil() == True:
            return None
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})

    def isnil(self):
        '''
        判断红黑树结点是否是哨兵结点
        '''
        if self.key == None and self.color == BLACK:
            return True
        return False

class RedBlackTree:
    '''
    红黑树
    '''
    def __init__(self):
        '''
        红黑树
        '''
        self.nil = self.buildnil()
        self.root = self.nil

    def buildnil(self):
        '''
        构造一个新的哨兵nil结点
        '''
        nil = RedBlackTreeNode(None, color=BLACK)
        return nil

    def insertkey(self, key, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = RedBlackTreeNode(key, index, color)
        self.insert(z)

    def successor(self, x : RedBlackTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h=lgn`为树的高度
        
        '''
        if x.right != self.nil:
            return self.minimum(x.right)
        y = x.p
        while y != self.nil and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : RedBlackTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != self.nil:
            return self.maximum(x.left)
        y = x.p
        while y != self.nil and x == y.left:
            x = y
            y = y.p
        return y

    def tree_search(self, x : RedBlackTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != self.nil and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return self.nil

    def minimum(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != self.nil:
            x = x.left
        return x

    def __minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def maximum(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != self.nil:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def insert(self, z : RedBlackTreeNode):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = RED
        self.insert_fixup(z)

    def insert_fixup(self, z : RedBlackTreeNode):
        '''
        插入元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while z.p.color == RED:
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)               
        self.root.color = BLACK    
        
    def delete_fixup(self, x : RedBlackTreeNode):
        '''
        删除元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while x != self.root and x.color == BLACK:
            if x == x.p.left:
                w : RedBlackTreeNode = x.p.right
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.leftrotate(x.p)
                    w = x.p.right
                elif w.color == BLACK:
                    if w.left.color == BLACK and w.right.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self.rightrotate(w)
                        w = x.p.right
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.right.color = BLACK
                        self.leftrotate(x.p)
                        x = self.root
            else:
                w : RedBlackTreeNode = x.p.left
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.rightrotate(x.p)
                    w = x.p.left
                elif w.color == BLACK:
                    if w.right.color == BLACK and w.left.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.leftrotate(w)
                        w = x.p.left
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.left.color = BLACK
                        self.rightrotate(x.p)
                        x = self.root
        x.color = BLACK

    def delete(self, z : RedBlackTreeNode):
        '''
        删除红黑树结点
        '''
        if z.isnil() == True:
            return
        if z.left == self.nil or z.right == self.nil:
            y = z
        else:
            y = self.successor(z)
        if y.left != self.nil:
            x = y.left
        else:
            x = y.right
        x.p = y.p
        if x.p == self.nil:
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        if y != z:
            z.key = y.key
            z.index = _deepcopy(y.index)
        if y.color == BLACK:
            self.delete_fixup(x)
        return y
    
    def deletekey(self, key):
        '''
        删除红黑树结点
        '''
        node = self.tree_search(self.root, key)
        return self.delete(node)

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋 时间复杂度: `O(1)`
        '''
        y : RedBlackTreeNode = x.right
        z = y.left
        if y == self.nil:
            return 
        y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        x.right = z

    def rightrotate(self, x : RedBlackTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        y : RedBlackTreeNode = x.left
        z = y.right
        if y == self.nil:
            return
        y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
        x.left = z
            
    def inorder_tree_walk(self, x : RedBlackTreeNode):
        '''
        从红黑树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None and x.isnil() == False:
            array.append(str(x))
            array = array + right
        return array
    
    def all(self):
        '''
        按`升序` 返回红黑树中所有的结点
        '''
        return self.inorder_tree_walk(self.root)

    def clear(self):
        '''
        清空红黑树
        '''
        self.destroy(self.root)
        self.root = self.buildnil()

    def destroy(self, x : RedBlackTreeNode):
        '''
        销毁红黑树结点
        '''
        if x == None:
            return
        if x.left != None:   
            self.destroy(x.left)
        if x.right != None:  
            self.destroy(x.right) 
        x = None
  
    def __preorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            print(str(node), end=' ')  
            self.__preorder(node.left) 
            self.__preorder(node.right)  

    def __inorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left) 
            print(str(node), end=' ') 
            self.__preorder(node.right)  

    def __postorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left)       
            self.__preorder(node.right) 
            print(str(node), end=' ') 

    def preorder_print(self):
        '''
        前序遍历红黑树
        ''' 
        print('preorder')
        self.__preorder(self.root)
        print('')

    def inorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('inorder')
        self.__inorder(self.root)
        print('')

    def postorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('postorder')
        self.__postorder(self.root)
        print('')

    @staticmethod
    def test():
        tree = RedBlackTree()
        tree.insertkey(41)
        tree.insertkey(38)
        tree.insertkey(31)
        tree.insertkey(12)
        tree.insertkey(19)
        tree.insertkey(8)
        tree.insertkey(1)
        tree.deletekey(12)
        tree.deletekey(38)
        tree.preorder_print()
        tree.postorder_print()
        tree.inorder_print()
        print(tree.all())
        tree.clear()
        print(tree.all())

if __name__ == '__main__':
    
    RedBlackTree.test()

    # python src/chapter13/redblacktree.py
    # python3 src/chapter13/redblacktree.py

else:
    pass
```

```py

# python src/chapter14/chapter14note.py
# python3 src/chapter14/chapter14note.py
'''
Class Chapter14_1

Class Chapter14_2

Class Chapter14_3

'''

from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import sys as _sys
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange

class Chapter14_1:
    '''
    chpater14.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.1 note

        Example
        ====
        ```python
        Chapter14_1().note()
        ```
        '''
        print('chapter14.1 note as follow')
        print('第14章 数据结构的扩张')
        print('在有些工程应用环境中，需要一些标准的数据结构，如双链表、散列表或二叉查找树')
        print('同时，也有许多应用要求在现有数据结构上有所创新，但很少需要长早出全新的数据结构')
        print('这一章讨论两种通过扩充红黑树构造的数据结构')
        print('14.1 动态顺序统计')
        print('第9章中介绍了顺序统计的概念。例如，在包含n个元素的集合中，第i个顺序统计量即为该集合中具有第i小关键字的元素')
        print('在一个无序的集合中，任意的顺序统计量都可以在O(n)时间内找到')
        print('这一节里，将介绍如何修改红黑树的结构，使得任意的顺序统计量都可以在O(lgn)时间内确定')
        print('还将看到，一个元素的排序可以同样地在O(lgn)时间内确定')
        print('一棵顺序统计量树T通过简单地在红黑树的每个结点存入附加信息而成',
            '在一个结点x内，除了包含通常的红黑树的域key[x],color[x],p[x],left[x]和right[x],还包括域size[x]')
        print('这个域中包含以结点x为根的子树的内部结点数(包括x本身)，即子树的大小，如果定义哨兵为0，也就是设置size[nil[T]]为0')
        print('则有等式size[x]=size[left[x]]+size[right[x]]+1')
        print('在一个顺序统计树中，并不要求关键字互不相同')
        print('在出现相等关键字的情况下，先前排序的定义不再适用。')
        print('定义排序为按中序遍历树时输出的结点位置，以此消除顺序统计树原定义的不确定性')
        print('OS-RANK的while循环的每一次迭代要花O(1)时间，且y在每次迭代中沿树上升一层')
        print(' 所以最坏情况下，OS-RANK的运行时间与树的高度成正比：对含n个结点的顺序统计树时间为O(lgn)')
        print('对子树规模的维护：给定每个结点的size域后，OS-SELECT和OS-RANK能迅速计算出所需的顺序统计信息')
        print('维护size域的代价为O(lgn)')
        print('红黑树上的插入操作包括两个阶段。第一个阶段从根开始，沿着树下降，将新结点插入为某个已有结点的孩子')
        print('第二阶段沿树上升，做一些颜色修改和旋转以保持红黑性质')
        print('于是，向一个含n个结点的顺序统计树中插入所需的总时间为O(lgn),从渐进意义上来看，这与一般的红黑树是相同的')
        print('红黑树上的删除操作同样包含两个阶段：第一阶段对查找树进行操作；第二阶段做至多三次旋转')
        print('综上所述,插入操作和删除操作，包括维护size域，都需O(lgn)时间')
        print('练习14.1-1: 略')
        print('练习14.1-2: 略')
        print('练习14.1-3: 完成')
        print('练习14.1-4: 完成')
        print('练习14.1-5: 给定含n个元素的顺序统计树中的一个元素x和一个自然数i，',
            '如何在O(lgn)时间内，确定x在该树的线性序中第i个后继')
        print('练习14.1-6: 在OS-SELECT或OS-RANK中，每次引用结点的size域都',
            '仅是为了计算在以结点为根的子树中该结点的rank')
        print('练习14.1-7: 说明如何在O(nlgn)的时间内，利用顺序统计树对大小为n的数组中的逆序对')
        print('练习14.1-8: 现有一个圆上的n条弦，每条弦都是按其端点来定义的。',
            '请给出一个能在O(nlgn)时间内确定园内相交弦的对数。假设任意两条弦都不会共享端点')      
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

class Chapter14_2:
    '''
    chpater14.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.2 note

        Example
        ====
        ```python
        Chapter14_2().note()
        ```
        '''
        print('chapter14.2 note as follow')
        print('14.2 如何扩张数据结构')
        print('对一种数据结构的扩张过程可分为四个步骤')
        print(' 1.选择基础数据结构')
        print(' 2.确定要在基础数据结构中添加哪些信息')
        print(' 3.验证可用基础数据结构上的基本修改操作来维护这些新添加的信息')
        print(' 4.设计新的操作')
        print('对红黑树的扩张')
        print('当红黑树被选作基础数据结构时，可以证明，',
            '某些类型的附加信息总是可以用插入和删除来进行有效地维护')
        print('定理14.1(红黑树的扩张) 设域f对含n个结点的红黑树进行扩张的域，',
            '且某结点x的域f的内容可以仅用结点x,left[x]和right[x]中的信息计算')
        print(' 包括f[left[x]]和f[right[x]]')
        print(' 在插入和删除操作中，我们在不影响这两个操作O(lgn)渐进性能的情况下，对T的所有结点的f值进行维护')
        print('练习14.2-1 通过为结点增加指针的形式可以在扩张的顺序统计树上，以最坏情况O(1)的时间来支持')
        print(' 动态集合查询操作MINIMUM,MAXIMUM,SUCCESSOR,PREDECESSOR,且顺序统计树上的其他操作的渐进性能不受影响')
        print('练习14.2-2 可以为结点增加黑高度域，且不影响红黑树操作性能')
        print('练习14.2-3 可以将红黑树中结点的深度作为一个域来进行有效的维护')
        print('练习14.2-4 假设在红黑树每个结点x中增加一个域f，按中序排列所有结点。证明在一次旋转后')
        print(' 可以在O(1)时间里对f域作出合适的修改。对扩张稍作修改，证明顺序统计树size域的每次旋转的维护时间为O(1)')
        print('练习14.2-5 希望通过增加操作RB-ENUMERATE(x,a,b)来扩张红黑树。该操作输出所有的关键字k')
        print(' 在Θ(m+lgn)时间里实现RB-ENUMERATE，其中m为输出的关键字数，n为树中内部结点(不需要向红黑树中增加新的域)')
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

class Chapter14_3:
    '''
    chpater14.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter14.3 note

        Example
        ====
        ```python
        Chapter14_3().note()
        ```
        '''
        print('chapter14.3 note as follow')
        print('14.3 区间树')
        print('定义一个闭区间是一个有序对[t1,t2]。开区间和半开区间分别略去了集合的两个或一个端点')
        print('区间可以很方便地表示各占用一段连续时间的一些事件')
        print('区间树是一种对动态集合进行维护的红黑树，该集合中的每个元素x都包含一个区间int[x]')
        print('区间树以及区间树上各种操作的设计')
        print(' 1.基础数据结构：红黑树；其中，每个结点x包含一个区间域int[x],x的关键字为区间的低端点low[int[x]]')
        print(' 2.附加信息：每个结点中除了区间信息外，还包含一个值max[x],即以x为根的子树中所有区间的端点的最大值')
        print(' 3.对信息的维护:必须验证对含n个结点的区间树的插入和删除能在O(lgn)时间内完成')
        print('  给定区间int[x]和x的子结点的max值，可以确定max[x]')
        print('     max[x]=max(x.high, x.left.max, x.right.max)')
        print('  这样根据定理14.1可知，插入和删除操作的运行时间O(lgn)。在一次旋转后，更新max域只需O(1)时间')
        print(' 4.设计新的操作:唯一需要的新操作是INTERVAL-SEARCH(T, i)')
        print('定理14.2 INTERVAL-SEARCH(T, i)的任意一次执行都将或者返回一个其区间覆盖了i的结点，',
            '或者返回nil[T],此时树T中没有哪一个结点的区间覆盖了i')
        print('练习14.3-1 写出作用于区间树的结点，并于O(1)时间内更新max域的LEFT-ROTATE的伪代码：完成')
        print('练习14.3-2 重写INTERVAL-SEAECH代码，使得当所有的区间都是开区间时，它也能正确的工作')
        print('练习14.3-3 请给出一个有效的算法，使对给定的区间i，它返回一个与i重叠的、',
            '具有最小低端点的区间；或者，当这样的区间不存在时返回nil[T]')
        print('练习14.3-4 给定一个区间树T和一个区间i，请描述如何能在O(min(n, klgn))时间内，',
            '列出T中所有与i重叠的区间，此处k为输出区间数。(可选：找出一种不修改树的方法)')
        print('练习14.3-5 请说明如何对有关区间树的的过程作哪些修改，才能支持操作INTERVAL-SEARCH-EXACTLY(T, i)',
            '它返回一个指向区间树T中结点x的指针，使low[int[x]]=low[i],high[int[x]]=high[i]',
            '或当T不包含这样的结点时返回nil[T],所有操作对于n结点树的运行时间应该为O(lgn)')
        print('练习14.3-6 请说明如何来维护一个支持操作MIN-GAP的动态数集Q，使该操作能给出Q中最近的两个数之间的差幅值',
            '例如Q={1,5,9,15,18,22}，则MIN-GAP(Q)返回18-15=3，因为15和18为Q最近的两数',
            '使操作INSERT,DELETE,SEARCH和MIN-GAP尽可能高效，并分析它们的运行时间')
        print('练习14.3-7 VLSI数据库通常将一块集成电路表示成一组矩形，因而矩形的表示中有最小和最大的x和y坐标')
        print(' 请给出一个能在O(lgn)时间里确定一组矩形中是否有两个重叠的算法(提示：将一条线移过所有的矩形)')
        print('思考题14-1 最大重叠点')
        print(' 假设希望对一组区间记录一个最大重叠点，亦即覆盖它的区间最多的那个点')
        print(' 1.证明：最大重叠点总存在于某段的端点上')
        print(' 2.设计一组数据结构，能有效的支持操作INTERVAL-INSERT,INTERVAL-DELETE和返回最大重叠点操作FIND-POM')
        print(' 提示：将所有端点组织成红黑树。左端点关联+1值，右端点关联-1值，附加一些维护最大重叠点信息以扩张树中结点')
        print('思考题14-2 Josephus排列')
        print(' Josephus问题的定义如下：假设n个人排成环形，且有一正整数m<=n。')
        print(' 从某个指定的人开始，沿环报数，每遇到第m个人就让其出列，且报数进行下去。这个过程一直进行到所有人都出列为止')
        print(' 每个人出列的次序定义了整数1,2,...,n的(n,m)-Josephus排列。例如(7,3)-Josephus排列为<3,6,2,7,5,1,4>')
        print(' 1.假设m为常数。请描述一个O(n)的算法，使之对给定的整数n，输出(n,m)-Josephus排列')
        print(' 2.假设m不是个常数，请描述一个O(nlgn)时间的算法，使给定的整数n和m，输出(n,m)-Josephus排列')
        # python src/chapter14/chapter14note.py
        # python3 src/chapter14/chapter14note.py

chapter14_1 = Chapter14_1()
chapter14_2 = Chapter14_2()
chapter14_3 = Chapter14_3()

def printchapter14note():
    '''
    print chapter14 note.
    '''
    print('Run main : single chapter fourteen!')  
    chapter14_1.note()
    chapter14_2.note()
    chapter14_3.note()

# python src/chapter14/chapter14note.py
# python3 src/chapter14/chapter14note.py
if __name__ == '__main__':  
    printchapter14note()
else:
    pass

```

```py


from __future__ import division, absolute_import, print_function
from copy import deepcopy as _deepcopy

BLACK = 0
RED = 1

class RedBlackTreeNode:
    '''
    红黑树结点
    '''
    def __init__(self, key, size = 0, index = None, color = RED, \
        p = None, left = None, right = None):
        '''
        红黑树树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.color = color
        self.size = size
        self.p = p

    def __str__(self):
        '''
        str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})
        '''
        if self.isnil() == True:
            return None
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})

    def isnil(self):
        '''
        判断红黑树结点是否是哨兵结点
        '''
        if self.key == None and self.color == BLACK:
            return True
        return False

class RedBlackTree:
    '''
    红黑树
    '''
    def __init__(self):
        '''
        红黑树
        '''
        self.nil = self.buildnil()
        self.root = self.nil

    def buildnil(self):
        '''
        构造一个新的哨兵nil结点
        '''
        nil = RedBlackTreeNode(None, color=BLACK, size=0)
        return nil

    def insertkey(self, key, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = RedBlackTreeNode(key, index, color)
        self.insert(z)

    def successor(self, x : RedBlackTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h=lgn`为树的高度
        
        '''
        if x.right != self.nil:
            return self.minimum(x.right)
        y = x.p
        while y != self.nil and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : RedBlackTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != self.nil:
            return self.maximum(x.left)
        y = x.p
        while y != self.nil and x == y.left:
            x = y
            y = y.p
        return y

    def tree_search(self, x : RedBlackTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != self.nil and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return self.nil

    def minimum(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != self.nil:
            x = x.left
        return x

    def __minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def maximum(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != self.nil:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def insert(self, z : RedBlackTreeNode):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = RED
        self.insert_fixup(z)

    def insert_fixup(self, z : RedBlackTreeNode):
        '''
        插入元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while z.p.color == RED:
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)               
        self.root.color = BLACK    
        
    def delete_fixup(self, x : RedBlackTreeNode):
        '''
        删除元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while x != self.root and x.color == BLACK:
            if x == x.p.left:
                w : RedBlackTreeNode = x.p.right
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.leftrotate(x.p)
                    w = x.p.right
                elif w.color == BLACK:
                    if w.left.color == BLACK and w.right.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self.rightrotate(w)
                        w = x.p.right
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.right.color = BLACK
                        self.leftrotate(x.p)
                        x = self.root
            else:
                w : RedBlackTreeNode = x.p.left
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.rightrotate(x.p)
                    w = x.p.left
                elif w.color == BLACK:
                    if w.right.color == BLACK and w.left.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.leftrotate(w)
                        w = x.p.left
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.left.color = BLACK
                        self.rightrotate(x.p)
                        x = self.root
        x.color = BLACK

    def delete(self, z : RedBlackTreeNode):
        '''
        删除红黑树结点
        '''
        if z.isnil() == True:
            return
        if z.left == self.nil or z.right == self.nil:
            y = z
        else:
            y = self.successor(z)
        if y.left != self.nil:
            x = y.left
        else:
            x = y.right
        x.p = y.p
        if x.p == self.nil:
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        if y != z:
            z.key = y.key
            z.index = _deepcopy(y.index)
        if y.color == BLACK:
            self.delete_fixup(x)
        return y
    
    def deletekey(self, key):
        '''
        删除红黑树结点
        '''
        node = self.tree_search(self.root, key)
        return self.delete(node)

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋 时间复杂度: `O(1)`
        '''
        y : RedBlackTreeNode = x.right
        z = y.left
        if y == self.nil:
            return 
        y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        x.right = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def rightrotate(self, x : RedBlackTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        y : RedBlackTreeNode = x.left
        z = y.right
        if y == self.nil:
            return
        y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
        x.left = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1
            
    def inorder_tree_walk(self, x : RedBlackTreeNode):
        '''
        从红黑树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None and x.isnil() == False:
            array.append(str(x))
            array = array + right
        return array
    
    def all(self):
        '''
        按`升序` 返回红黑树中所有的结点
        '''
        return self.inorder_tree_walk(self.root)

    def clear(self):
        '''
        清空红黑树
        '''
        self.destroy(self.root)
        self.root = self.buildnil()

    def destroy(self, x : RedBlackTreeNode):
        '''
        销毁红黑树结点
        '''
        if x == None:
            return
        if x.left != None:   
            self.destroy(x.left)
        if x.right != None:  
            self.destroy(x.right) 
        x = None
  
    def __preorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            print(str(node), end=' ')  
            self.__preorder(node.left) 
            self.__preorder(node.right)  

    def __inorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left) 
            print(str(node), end=' ') 
            self.__preorder(node.right)  

    def __postorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left)       
            self.__preorder(node.right) 
            print(str(node), end=' ') 

    def preorder_print(self):
        '''
        前序遍历红黑树
        ''' 
        print('preorder')
        self.__preorder(self.root)
        print('')

    def inorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('inorder')
        self.__inorder(self.root)
        print('')

    def postorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('postorder')
        self.__postorder(self.root)
        print('')

    @staticmethod
    def test():
        tree = RedBlackTree()
        tree.insertkey(41)
        tree.insertkey(38)
        tree.insertkey(31)
        tree.insertkey(12)
        tree.insertkey(19)
        tree.insertkey(8)
        tree.insertkey(1)
        tree.deletekey(12)
        tree.deletekey(38)
        tree.preorder_print()
        tree.postorder_print()
        tree.inorder_print()
        print(tree.all())
        tree.clear()
        print(tree.all())

class OSTreeNode(RedBlackTreeNode):
    '''
    顺序统计树结点
    '''
    def __init__(self, key):
        '''
        `key` : 键值
        '''
        super().__init__(key)

    def __str__(self):
        '''
        str(OSTreeNode())

        {
            'key' : self.key, 
            'index' : self.index, 
            'color' : self.color,
            'size' : self.size
        }

        '''
        if self.isnil() == True:
            return None
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color,
            'size' : self.size})

class OSTree(RedBlackTree):
    '''
    顺序统计树
    '''
    def __init__(self):
        '''
        顺序统计树
        '''
        super().__init__()   

    def insert(self, z : OSTreeNode):
        '''
        插入顺序统计树结点
        '''
        super().insert(z)
        self.updatesize()

    def insertkey(self, key):
        '''
        插入顺序统计树结点
        '''
        node = OSTreeNode(key)
        self.insert(node)

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋 时间复杂度: `O(1)`
        '''
        y : RedBlackTreeNode = x.right
        z = y.left
        if y == self.nil:
            return 
        y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        x.right = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def rightrotate(self, x : RedBlackTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        y : RedBlackTreeNode = x.left
        z = y.right
        if y == self.nil:
            return
        y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
        x.left = z
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def __os_select(self, x : RedBlackTreeNode, i):
        r = x.left.size + 1
        if i == r:
            return x
        elif i < r:
            return self.__os_select(x.left, i)
        else:
            return self.__os_select(x.right, i - r)

    def os_select(self, i):
        '''
        返回树中包含第`i`小关键字的结点的指针(递归)
        '''
        assert i >= 1 
        return self.__os_select(self.root, i)

    def os_select_nonrecursive(self, i):
        '''
        返回树中包含第`i`小关键字的结点的指针(递归)
        '''
        r = -1
        x = self.root
        while i != r:
            last = x 
            r = x.left.size + 1
            if i < r:
                x = x.left
            elif i > r:
                x = x.right
                i = i - r
        return last

    def os_rank(self, x : RedBlackTreeNode):
        '''
        对顺序统计树T进行中序遍历后得到的线性序中`x`的位置
        '''
        r = x.left.size + 1
        y = x
        while y != self.root:
            if y == y.p.right:
                r = r + y.p.left.size + 1
            y = y.p
        return r    

    def os_key_rank(self, key):
        '''
        对顺序统计树T进行中序遍历后得到的线性序中键值为`key`结点的位置
        '''
        node = self.tree_search(self.root, key)
        return self.os_rank(node)

    def __updatesize(self, x : RedBlackTreeNode):
        if x.isnil() == True:
            return 0
        x.size = self.__updatesize(x.left) + self.__updatesize(x.right) + 1
        return x.size

    def updatesize(self):
        '''
        更新红黑树的所有结点的size域
        '''
        self.__updatesize(self.root)

    @staticmethod
    def test():
        '''
        测试函数
        '''
        tree = OSTree()
        tree.insertkey(12)
        tree.insertkey(13)
        tree.insertkey(5)
        tree.insertkey(8)
        tree.insertkey(16)
        tree.insertkey(3)
        tree.insertkey(1)    
        tree.insertkey(2)
        print(tree.all())
        print(tree.os_select(1))
        print(tree.os_select(2))
        print(tree.os_select(3))
        print(tree.os_select(4))
        print(tree.os_select(5))
        print(tree.os_select(6))
        print(tree.os_key_rank(8))
        print(tree.os_key_rank(12))

__para_interval_err = 'para interval must be a tuple like ' + \
            'contains two elements, min <= max'

class IntervalTreeNode(RedBlackTreeNode):
    '''
    区间树结点
    '''
    def __init__(self, key, interval : tuple):
        '''
        区间树结点

        `key` : 键值

        `interval` : 区间值 a tuple like (`min`, `max`), and `min` <= `max`

        '''

        try:
            assert type(interval) is tuple
            assert len(interval) == 2 
            self.low, self.high = interval
            assert self.low <= self.high 
        except:
            raise Exception(__para_interval_err)           
        super().__init__(key)      
        self.interval = interval 
        self.max = 0
    
    def __str__(self):
        '''
        {'key' : self.key, 
            'index' : self.index, 
            'color' : self.color,
            'interval' : self.interval }
        '''
        if self.isnil() == True:
            return 'None'
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color,
            'interval' : self.interval })

class IntervalTree(RedBlackTree):
    '''
    区间树
    '''
    def __init__(self):
        '''
        区间树
        '''
        super().__init__()
        
    def __updatemax(self, x : IntervalTreeNode):
        if x == None:
            return 0
        x.max = max(x.high, self.__updatemax(x.left), \
            self.__updatemax(x.right))
        return x.max
    
    def updatemax(self):
        '''
        更新区间树的`max`域
        '''
        self.__updatemax(self.root)

    def buildnil(self):
        '''
        构造一个新的哨兵nil结点
        '''
        nil = IntervalTreeNode(None, (0, 0))
        nil.color = BLACK
        nil.size = 0
        return nil

    def __int_overlap(self, int : tuple, i):
        low, high = int
        i = (i[0] + i[1]) / 2.0
        if i >= low and i <= high:
            return True
        return False

    def insert(self, x : IntervalTreeNode):
        '''
        将包含区间域`int`的元素`x`插入到区间树T中
        '''
        super().insert(x)
        self.updatemax()

    def delete(self, x : IntervalTreeNode):
        '''
        从区间树中删除元素`x`
        '''
        super().delete(x)
        self.updatemax()

    def interval_search(self, interval):
        '''
        返回一个指向区间树T中的元素`x`的指针，使int[x]与i重叠，
        若集合中无此元素存在，则返回`self.nil`
        '''
        x = self.root
        while x.isnil() == False and self.__int_overlap(x.interval, interval) == False:
            if x.left.isnil() == False and x.left.max >= interval[0]:
                x = x.left
            else:
                x = x.right
        return x

    def insertkey(self, key, interval, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = IntervalTreeNode(key, interval)
        self.insert(z)

    @staticmethod
    def test():
        '''
        test
        '''
        tree = IntervalTree()
        tree.insertkey(11, (0, 11))
        tree.insertkey(23, (11, 23))
        tree.insertkey(13, (12, 13))
        tree.insertkey(41, (40, 41))
        tree.insertkey(22, (11, 22))
        tree.insertkey(53, (42, 53))
        tree.insertkey(18, (10, 18))
        tree.insertkey(32, (22, 32))
        node = IntervalTreeNode(2, (1, 2))
        print(tree.interval_search((20, 25)))
        print(tree.interval_search((28, 33)))
        print(tree.all())

if __name__ == '__main__': 
    RedBlackTree.test()
    OSTree.test()
    IntervalTree.test()
    # python src/chapter14/rbtree.py
    # python3 src/chapter14/rbtree.py
else:
    pass


```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
'''
Class Chapter15_1

Class Chapter15_2

Class Chapter15_3

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import *

import io
import sys 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 

class Chapter15_1:
    '''
    chpater15.1 note and function
    '''
    index1 = 0
    index2 = 1
    f = [[], []]
    l = [[], []]
    def fastway(self, a, t, e, x, n):
        '''
        计算最快时间 Θ(n)

        Args
        ===
        `a` : `a[i][j]` 表示在第`i`条装配线`j`装配站的装配时间

        `t` : `t[i][j]` 表示在第`i`条装配线`j`装配站移动到另外一条装配线所需要的时间

        `e` : `e[i]` 表示汽车底盘进入工厂装配线`i`所需要的时间

        `x` : `x[i]` 表示完成的汽车花费离开装配线所需要的时间

        `n` : 每条装配线所具有的装配站数量

        Return
        ===
        `(fxin, lxin)` : a tuple like

        Example
        ===
        ```python
        a = [[7, 9, 3, 4, 8, 4], [8, 5, 6, 4, 5, 7]]
        t = [[2, 3, 1, 3, 4], [2, 1, 2, 2, 1]]
        e = [2, 4]
        x = [3, 2]
        n = 6
        self.fastway(a, t, e, x, n)
        >>> (38, 0)
        ```

        '''
        # 定义最优解变量
        ## 路径最优解
        lxin = 0
        ## 时间最优解
        fxin = 0
        # 定义两条装配线
        index1 = self.index1
        index2 = self.index2
        # 子问题存储空间
        f = self.f
        l = self.l
        # 开辟空间存储动态规划子问题的解
        f[index1] = list(range(n))
        f[index2] = list(range(n))
        l[index1] = list(range(n))
        l[index2] = list(range(n))
        # 上装配线
        f[index1][0] = e[index1] + a[index1][0]
        f[index2][0] = e[index2] + a[index2][0]
        # 求解子问题
        for j in range(1, n):
            # 求解装配线1的子问题,因为求解最短时间，谁小赋值谁
            if f[index1][j - 1] + a[index1][j] <= f[index2][j - 1] + t[index2][j - 1] + a[index1][j]:
                f[index1][j] = f[index1][j - 1] + a[index1][j]
                l[index1][j] = index1
            else:
                f[index1][j] = f[index2][j - 1] + t[index2][j - 1] + a[index1][j]
                l[index1][j] = index2
            # 求解装配线1的子问题,因为求解最短时间，谁小赋值谁
            if f[index2][j - 1] + a[index2][j] <= f[index1][j - 1] + t[index1][j - 1] + a[index2][j]:
                f[index2][j] = f[index2][j - 1] + a[index2][j]
                l[index2][j] = index2
            else:
                f[index2][j] = f[index1][j - 1] + t[index1][j - 1] + a[index2][j]
                l[index2][j] = index1
        n = n - 1
        # 求解离开装配线时的解即为总问题的求解，因为子问题已经全部求解
        if f[index1][n] + x[index1] <= f[index2][n] + x[index2]:
            fxin = f[index1][n] + x[index1]
            lxin = index1
        else:
            fxin = f[index2][n] + x[index2]
            lxin = index2
        # 返回最优解
        return (fxin, lxin)

    def printstations(self, l, lxin, n):
        '''
        打印最优通过的路线
        '''
        index1 = self.index1
        index2 = self.index2
        i = lxin - 1
        print('line', i + 1, 'station', n)
        for j in range(2, n + 1):
            m = n - j + 2 - 1
            i = l[i][m]
            print('line', i + 1, 'station', m)

    def __printstations_ascending(self, l, i, m):
        if m - 1 <= 0:
            print('line', i + 1, 'station', m)
        else:
            self.__printstations_ascending(l, l[i][m - 1], m - 1)
        print('line', i + 1, 'station', m)
        
    def printstations_ascending(self, l, lxin, n):
        '''
        升序打印最优通过的路线(递归方式)
        '''
        index1 = self.index1
        index2 = self.index2
        _lxin = lxin - 1
        self.__printstations_ascending(l, _lxin, n)

    def note(self):
        '''
        Summary
        ====
        Print chapter15.1 note

        Example
        ====
        ```python
        Chapter15_1().note()
        ```
        '''
        print('chapter15.1 note as follow')   
        print('第四部分 高级设计和分析技术')
        print('这一部分将介绍设计和分析高效算法的三种重要技术：动态规划(第15章)，贪心算法(第16章)和平摊分析(第17章)')
        print('本书前面三部分介绍了一些可以普遍应用的技术，如分治法、随机化和递归求解')
        print('这一部分的新技术要更复杂一些，但它们对有效地解决很多计算问题来说很有用')
        # !动态规划适用于问题可以分解为若干子问题,关键技术是存储这些子问题每一个解，以备它重复出现
        print('动态规划通常应用于最优化问题，即要做出一组选择以达到一个最优解。',
            '在做选择的同时，经常出现同样形式的子问题。当某一特定的子问题可能出自于多于一种选择的集合时，动态规划非常有效')
        print(' 关键技术是存储这些子问题每一个解，以备它重复出现。第15章说明如何利用这种简单思想，将指数时间的算法转化为多项式时间的算法')
        print('像动态规划算法一样，贪心算法通常也是应用于最优化问题。在这种算法中，要做出一组选择以达到一个最优解。')
        print(' 采用贪心算法可以比用动态规划更快地给出一个最优解。但是不同意判断贪心算法是否一定有效。')
        print(' 第16章回顾拟阵理论，它通常可以用来帮助做出这种判断。')
        print('平摊分析是一种用来分析执行一系列类似操作的算法的工具。',
            '平摊分析不仅仅是一种分析工具，也是算法设计的一种思维方式，',
                '因为算法的设计和对其运行时间的分析经常是紧密相连的')
        print('第15章 动态规划')
        print('和分治法一样，动态规划是通过组合子问题的解而解决整个问题的')
        print('分治法算法是指将问题划分成一些独立的子问题，递归地求解各子问题，然后合并子问题的解而得到原问题的解')
        print('于此不同，动态规划适用于子问题不是独立的情况，也就是各子问题包含的公共的子子问题。')
        print('动态规划不需要像分治法那样重复地求解子子问题，对每个子子问题只求解一次，将其结果保存在一张表中')
        print('动态规划通常应用于最优化问题。此类问题可能有很多种可行解。每个解有一个值，希望找出一个具有最优(最大或最小)值的解')
        print('动态规划算法的设计可以分为如下4个步骤：')
        print(' 1.描述最优解的结构')
        print(' 2.递归定义最优解的值')
        print(' 3.按自底向上的方式计算最优解的值')
        print(' 4.由计算出的结果构造一个最优解')
        print('第1~3步构成问题的动态规划解的基础。第4步在只要求计算最优解的值时可以略去')
        print('15.1 装配线调度')
        print('一个动态规划的例子是求解一个制造问题')
        print('某汽车公司在有两条装配线的工厂内生产汽车，一个汽车底盘在进入每一条装配线后，在一些装配站中会在底盘上安装部件')
        print('然后，完成的汽车在装配线的末端离开。每一条装配线上有n个装配站，编号为j=1,2..,n')
        print('将装配线i(i为1或2)的第j个装配站表示为Si,j。装配线1的第j个站(S1,j)和装配线2的第j个站(S2,j)执行相同的功能')
        print('然而，这些装配站是在不同的时间建造的，并且采用了不同的技术；因此在每个站上所需的时间是不同的')
        print('在不同站所需要的时间为aij,一个汽车底盘进入工厂，然后进入装配线i(i为1或2),花费时间ei.')
        print('在通过一条线的第j个装配站后，这个底盘来到任一条线的第(j+1)个装配站')
        print('如果它留在相同的装配线，则没有移动的开销；但是，如果在装配站Sij后，它移动了另一条线上，则花费时间为tij')
        print('在离开一条线的第n个装配站后，完成的汽车花费时间xi离开工厂。待求解的问题是确定应该在装配线1内选择哪些站、在装配线2内选择哪些站')
        print('才能使汽车通过工厂的总时间最小')
        print('显然，当有很多个装配站时，用强力法(brute force)来极小化通过工厂装配线的时间是不可能的。')
        print('如果给定一个序列，在装配线1上使用哪些站，在装配线2上使用哪些站，则可以在Θ(n)时间内,',
            '很容易计算出一个底盘通过工厂装配线要花的时间')
        print('不幸地是，选择装配站的可能方式有2^n种；可以把装配线1内使用的装配站集合看作{1,2,..,n}的一个子集')
        print('因此，要通过穷举所有可能的方式、然后计算每种方式花费的时间来确定最快通过工厂的路线，需要Ω(2^n)时间，这在n很大时是不行的')
        print('步骤1.通过工厂最快路线的结构,子问题最优结果结果的存储空间')
        print(' 动态规划方法的第一个步骤是描述最优解的结构的特征。对于装配线调度问题，可以如下执行。')
        print(' 首先，假设通过装配站S1,j的最快路线通过了装配站S1,j-1。关键的一点是这个底盘必定是利用了最快的路线从开始点到装配站S1,j-1的')
        print(' 更一般地，对于装配线调度问题，一个问题的最优解包含了子问题的一个最优解。')
        print(' 我们称这个性质为最优子结构，这是是否可以应用动态规划方法的标志之一')
        print(' 为了寻找通过任一条装配线上的装配站j的最快路线，我们解决它的子问题，即寻找通过两条装配线上的装配站j-1的最快路线')
        print(' 所以，对于装配线调度问题，通过建立子问题的最优解，就可以建立原问题某个实例的一个最优解')
        print('步骤2.一个递归的解，总时间最快就是子问题最快')
        print(' 在动态规划方法中，第二个步骤是利用子问题的最优解来递归定义一个最优解的值。')
        print(' 对于装配线的调度问题，选择在两条装配线上通过装配站j的最快路线的问题来作为子问题')
        print(' j=1,2,...,n。令fi[j]表示一个底盘从起点到装配站Sij的最快可能时间')
        print(' 最终目标是确定底盘通过工厂的所有路线的最快时间，记为f')
        print(' 底盘必须一路径由装配线1或2通过装配站n，然后到达工厂的出口，由于这些路线的较快者就是通过整个工厂的最快路线')
        print(' f=min(f1[n]+x1,f2[n]+x2)')
        print(' 要对f1[1]和f2[1]进行推理也是容易的。不管在哪一条装配线上通过装配站1，底盘都是直接到达该装配站的')
        print('步骤3.计算最快时间fxin')
        print(' 此时写出一个递归算法来计算通过工厂的最快路线是一件简单的事情，',
            '这种递归算法有一个问题：它的执行时间是关于n的指数形式')
        # !装配站所需时间
        a = [[7, 9, 3, 4, 8, 4],\
             [8, 5, 6, 4, 5, 7]]
        # !装配站切换到另一条线花费的时间
        t = [[2, 3, 1, 3, 4],\
             [2, 1, 2, 2, 1]]
        # !进入装配线所需时间
        e = [2, 4]
        # !离开装配线所需时间
        x = [3, 2]
        # !每条装配线装配站的数量
        n = 6
        result = self.fastway(a, t, e, x, n)
        fxin = result[0]
        lxin = result[1] + 1
        print('fxin:', fxin, ' lxin:', lxin, 'l[lxin]:', _array(self.l)[lxin - 1] + 1)
        print('最优路径输出(降序从尾到头)')
        self.printstations(self.l, lxin, n)    
        print('存储的子问题的解为：')
        print('f:')
        print(_array(self.f))
        print('l:')
        print(_array(self.l) + 1)
        print('步骤4.构造通过工厂的最快路线lxin')
        print('练习15.1-1 最优路径输出(升序从头到尾)')
        # 通过递归的方式先到达路径头
        self.printstations_ascending(self.l, lxin, n)
        print('练习15.1-2 定理：在递归算法中引用fi[j]的次数ri(j)等于2^(n-j)')
        print('练习15.1-3 定理：所有引用fi[j]的总次数等于2^(n+1)-2')
        print('练习15.1-4 包含fi[j]和li[j]值的表格共含4n-2个表项。说明如何把空间需求缩减到共2n+2')
        print('练习15.1-5 略')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_2:
    '''
    chpater15.2 note and function
    '''

    def matrix_multiply(self, A, B):
        '''
        两个矩阵相乘
        '''
        rowA = shape(A)[0]
        colunmA = shape(A)[1]
        rowB = shape(B)[0]
        colunmB = shape(B)[1]
        C = ones([rowA, colunmB])
        if colunmA != rowA:
            raise Exception('incompatible dimensions')
        else:
            for i in range(rowA):
                for j in range(colunmB):
                    C[i][j] = 0
                    for k in range(colunmA):
                        C[i][j] = C[i][j] + A[i][k] * B[k][j]
            return C

    def matrix_chain_order(self, p):
        '''
        算法：填表`m`的方式对应于求解按长度递增的矩阵链上的加全部括号问题

        Return
        ===
        `(m, s)`

        `m` : 存储子问题的辅助表`m`

        `s` : 存储子问题的辅助表`s`

        Example
        ===
        ```python
        matrix_chain_order([30, 35, 15, 5, 10, 20, 25])
        >>> (m, s)
        ```
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        for i in range(n):
            m[i][i] = 0
        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l - 1
                m[i][j] = math.inf
                for k in range(i, j):
                    q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]          
                    if q < m[i][j]:
                        m[i][j] = q
                        s[i][j] = k + 1
        return (m, s)

    def __print_optimal_parens(self, s, i, j):
        '''
        输出矩阵链乘积的一个最优加全部括号形式
        '''
        i = int(i)
        j = int(j)
        if i == j:
            print('A{}'.format(i + 1), end='')
        else:
            print('(', end='')
            self.__print_optimal_parens(s, i, s[i][j])
            self.__print_optimal_parens(s, s[i][j] + 1, j)
            print(')', end='')

    def print_optimal_parens(self, s):
        '''
        输出矩阵链乘积的一个最优加全部括号形式
        '''
        s = s - 1
        self.__print_optimal_parens(s, 0, shape(s)[-1] - 1)

    def __matrix_chain_multiply(self, A, s, i, j):
        pass

    def matrix_chain_multiply(self, A):
        '''
        调用矩阵链乘法对矩阵数组进行连乘
        '''
        p = []
        for a in A:
            row = shape(a)[0]
            p.append(row)
        p.append(shape(A[-1])[1])
        m, s = self.matrix_chain_order(p)
        return self.__matrix_chain_multiply(A, s, 1, len(p) - 1)

    def note(self):
        '''
        Summary
        ====
        Print chapter15.2 note

        Example
        ====
        ```python
        Chapter15_2().note()
        ```
        '''
        print('chapter15.2 note as follow')   
        print('15.2 矩阵链乘法')
        print('例如：如果矩阵链为A B C D')
        print('则乘积ABCD可用五种不同的方式加括号')
        print('(A(B(CD)))')
        print('(A((BC)D))')
        print('((AB)(CD))')
        print('((A(BC))D)')
        print('(((AB)C)D)')
        A = [[1, 2], [3, 4]]
        print(self.matrix_multiply(A, A))
        A = array(A)
        print(A * A)
        print('为了计算矩阵链乘法，可将两个矩阵相乘的标准算法作为一个子程序Θ(n^3)，矩阵乘法满足结合律')
        print('矩阵链乘法加括号的顺序对求积运算的代价有很大的影响。')
        print('矩阵乘法当且仅当两个矩阵相容(A的列数等于B的行数)，才可以进行相乘运算')
        print('C(p,r) = A(p,q) * B(q,r)')
        print('注意在矩阵链乘法当中，实际上并没有把矩阵相乘，目的仅是确定一个具有最下代价的相乘顺序')
        print('确定最优顺序花费的时间能在矩阵乘法上得到更好的回报')
        print('计算全部括号的重数')
        print('设P(n)表示一串n个矩阵可能的加全部括号的方案数')
        print('用动态规划求解矩阵链乘法顺序，使用穷举的方式不是很好的一个方式，随着矩阵数量n的增长，')
        print('P(n)的一个类似递归解的Catalan数序列，其增长的形式是Ω(4^n/n^(3/2))')
        print('P(n)递归式的一个解为Ω(2^n),所以解的个是指数形式，穷尽策略不是一个好的形式')
        print('步骤1.最优加全部括号的结构')
        print(' 动态规划方法的第一步是寻找最优的子结构')
        print(' 然后，利用这一子结构，就可以根据子问题的最优解构造出原问题的一个最优解。')
        print(' 对于矩阵链乘法问题，可以执行如下这个步骤')
        print(' 用记号Ai..j表示对乘积AiAi+1Aj求值的结果，其中i<=j,如果这个问题是非平凡的，即i<j')
        print(' 则对乘积AiAi+1...Aj的任何全部加括号形式都将乘积在Ak与Ak+1之间分开，此处k是范围1<=k<j之内的一个整数')
        print(' 就是说，对某个k值，首先计算矩阵Ai..k和Ak+1..j,然后把它们相乘就得到最终乘积Ai..j')
        print(' 这样，加全部括号的代价就是计算Ai..k和Ak+1..j的代价之和再加上两者相乘的代价')
        print('步骤2.一个递归解')
        print(' 接下来，根据子问题的最优解来递归定义一个最优解的代价。对于矩阵链乘法问题，子问题即确定AiAi+1...Aj的加全部括号的最小代价问题')
        print(' 此处1<=i<=j<=n。设m[i,j]为计算矩阵Ai..j所需的标量乘法运算次数的最小值；对整个问题，计算A1..n的最小代价就是m[1,n]')
        print(' 递归定义m[i,j]。如果i==j,则问题是平凡的；矩阵链只包含一个矩阵Ai..i=Ai,故无需做任何标量乘法来计算chengji')
        print(' 关于对乘积AiAi+1...Aj的加全部括号的最小代价的递归定义为')
        print(' m[i,j]=0, i = j;  m[i,j]=min{min[i,k]+m[k+1,j] + pi-1pkpj}, i < j')
        print('步骤3.计算最优代价')
        print(' 可以很容易地根据递归式，来写一个计算乘积A1A2...An的最小代价m[1,n]的递归算法。',
            '然而这个算法具有指数时间，它与检查每一种加全部括号乘积的强力法差不多')
        print(' 但是原问题只有相当少的子问题：对每一对满足1<=i<=j<=n的i和j对应一个问题，总共Θ(n^2)种')
        print(' 一个递归算法在其递归树的不同分支中可能会多次遇到同一个子问题，子问题重叠这一性质')
        print(' 不是递归地解递归式，而是执行动态规划方法的第三个步骤，使用自底向上的表格法来计算最优代价')
        print(' 假设矩阵Ai的维数是pi-1×pi,i=1,2,...,n。输入是一个序列p=<p0,p1,...pn>,其中length[p]=n+1')
        print(' 程序使用一个辅助表m[1..n,1..n]来保存m[i,j]的代价')
        print('例子：矩阵链 A1(30 * 35) A2(35 * 15) A3(15 * 5) A4(5 * 10) A5(10 * 20) A6(20 * 25)')
        print('的一个最优加全部括号的形式为((A1(A2A3))((A4A5)A6))')
        p = [30, 35, 15, 5, 10, 20, 25]
        n = len(p) - 1
        m, s = self.matrix_chain_order(p)
        print('the m is ')
        print(m)
        print('the s is ')
        print(s)
        print('最优加全部括号形式为：')
        self.print_optimal_parens(s)
        print('')
        # self.print_optimal_parens(s, 0, n - 1)
        print('步骤4.构造一个最优解')
        print(' 虽然MATRIX—CHAIN-ORDER确定了计算矩阵链乘积所需的标量乘积法次数，但没有说明如何对这些矩阵相乘(如何加全部括号)')
        print(' 利用保存在表格s[1..n,1..n]内的、经过计算的信息来构造一个最优解并不难。')
        print(' 在每一个表项s[i,j]中，记录了对乘积AiAi+1...Aj在Ak与Ak+1之间，进行分裂以取得最优加全部括号时的k值')
        print('练习15.2-1 对6个矩阵维数为<5, 10, 3, 12, 5, 50, 6>的各矩阵，找出其矩阵链乘积的一个最优加全部括号')
        p = [5, 10, 3, 12, 5, 50, 6]
        n = len(p) - 1
        m, s = self.matrix_chain_order(p)
        print('the m is ')
        print(m)
        print('the s is ')
        print(s)
        print('最优加全部括号形式为：')
        self.print_optimal_parens(s)
        print('')
        print('练习15.2-2 给出一个矩阵链乘法算法MATRX-CHAIN_MULTIPLY(A, s, i, j), 初始参数为A, s, 1, n')
        print('练习15.2-3 用替换法证明递归公式的解为Ω(2^n)')
        print('练习15.2-4 设R(i, j)表示在调用MATRIX-CHAIN—ORDER中其他表项时，',
            '表项m[i, j]被引用的次数(n^3-n)/3')
        print('练习15.2-5 定理：一个含n个元素的表达式的加全部括号中恰有n-1对括号(显然n个数的乘法做n-1次两数相乘即可出结果)')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_3:
    '''
    chpater15.3 note and function
    '''
    def __recursive_matrix_chain(self, p, m, s, i, j):
        '''
        矩阵链算法的低效递归版本
        '''
        if i == j:
            return 0
        m[i][j] = math.inf
        for k in range(i, j):
            q = self.__recursive_matrix_chain(p, m, s, i, k) + self.__recursive_matrix_chain(p, m, s, k + 1, j) + p[i] * p[k + 1] * p[j + 1] 
            if q < m[i][j]:
                m[i][j] = q
                s[i][j] = k + 1
        return m[i, j]
        
    def recursive_matrix_chain(self, p):
        '''
        矩阵链算法的低效递归版本
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        self.__recursive_matrix_chain(p, m, s, 0, n - 1)
        return (m, s)

    def memoized_matrix_chain(self, p):
        '''
        矩阵链算法的备忘录版本
        '''
        # 矩阵的个数
        n = len(p) - 1
        # 辅助表m n * n
        m = zeros((n, n))
        # 辅助表s n * n
        s = zeros((n, n))
        # !备忘录版本与递归版本相同的地方都是要填表时进行递归，
        # !但是递归时并不重新计算表m中的元素,仅仅做一个某位置是否填过表的判断
        # 将表m全部填成无穷inf
        for i in range(n):
            for j in range(i, n):
                m[i][j] = math.inf
        self.loockup_chian(p, m, 0, n - 1)
        return m

    def loockup_chian(self, p, m, i, j):
        '''
        回溯查看表m中的元素
        '''
        # 查看而不是重新比较
        if m[i][j] < math.inf:
            return m[i][j]
        if i == j:
            m[i][j] = 0
        else:
            for k in range(i, j):
                q = self.loockup_chian(p, m, i, k) + \
                    self.loockup_chian(p, m, k + 1, j) + \
                    p[i] * p[k + 1] * p[j + 1] 
                if q < m[i][j]:
                    m[i][j] = q
        return m[i][j]

    def note(self):
        '''
        Summary
        ====
        Print chapter15.3 note

        Example
        ====
        ```python
        Chapter15_3().note()
        ```
        '''
        print('chapter15.3 note as follow')   
        print('15.3 动态规划基础')
        print('从工程的角度看，什么时候才需要一个问题的动态规划解')
        print('适合采用动态规划解决最优化问题的两个要素：最优子结构和重叠子问题')
        print('用备忘录充分利用重叠子问题性质')
        # !动态规划算法第一步寻找最优子结构，第二步递归的定义最优解的值
        print('最优子结构')
        print(' 用动态规划求解优化问题的第一步是描述最优解的结构')
        print(' 如果一个问题的最优解中包含了子问题的最优解，则该问题具有最优子结构')
        print(' 当一个问题包含最优子结构时，提示我们动态规划是可行的，当然贪心算法也是可行的')
        print('在寻找最优子结构时，可以遵循一种共同的模式')
        print(' 1.问题的一个解可以看作是一个选择')
        print(' 2.假设对一个给定的问题，已知的是一个可以导致最优解的选择')
        print(' 3.在已知这个选择后，要确定哪些子问题会随之发生')
        print(' 4.假设每个子问题的解都不可能是最优的选择，则问题也不可能是最优的')
        p = [5, 10, 3, 12, 5, 50, 6]
        n = len(p) - 1
        m, s = self.recursive_matrix_chain(p)
        print('the m is ')
        print(m)
        print('the s is ')
        print(s)
        print('the m is as follows:')
        print(self.memoized_matrix_chain(p))
        print('为了描述子问题空间，尽量保持这个空间简单')
        print('非正式地，一个动态规划算法地运行时间依赖于两个因素地乘积，子问题地总个数和每一个问题有多少种选择')
        print('在装配线调度中，总共有Θ(n)个子问题，并且只有两个选择来检查每个子问题，所以执行时间为Θ(n)。')
        print('对于矩阵链乘法，总共有Θ(n^2)个子问题，在每个子问题中又至多有n-1个选择，因此执行时间为O(n^3)')
        print('动态规划以自底向上的方式来利用最优子结构。首先找到子问题的最优解，解决子问题，然后找到问题的一个最优解')
        print('而贪心算法与动态规划有着很多相似之处。特别地，贪心算法适用的问题也具有最优子结构。')
        # !贪心算法与动态规划有一个显著的区别，就是在贪心算法中，是以自顶向下的方式使用最优子结构
        # !贪心算法会先做选择，在当时看起来是最优的选择，然后再求解一个结果子问题，而不是先寻找子问题的最优解，然后再做选择
        print('贪心算法与动态规划有一个显著的区别，就是在贪心算法中，是以自顶向下的方式使用最优子结构')
        print('贪心算法会先做选择，在当时看起来是最优的选择，然后再求解一个结果子问题，而不是先寻找子问题的最优解，然后再做选择')
        print('注意：在不能应用最优子结构的时候，就一定不能假设它能够应用，已知一个有向图G=(V,E)和结点u,v∈V')
        print('无权最短路径：找出一条从u到v的包含最少边数的路径。这样一条路径必须是简单路径，因为从路径中去掉一个回路后，会产生边数更少的路径')
        print('无权最长简单路径：找出一条从u到v的包含最多边数的简单路径，需要加入简单性需求，否则就可以遍历一个回路任意多次')
        print('这样任何从u到v的路径p必定包含一个中间顶点，比如w，')
        print('对无权最长简单路径问题，假设它具有最优子结构。最终结论：说明对于最长简单路径，不仅缺乏最优子结构，而且无法根据子问题的解来构造问题的一个合法解')
        print('而且在寻找最短路径中子问题是独立的，答案是子问题本来就没有共享资源')
        print('装配站问题和矩阵链乘法问题都有独立的子问题')
        print('重叠子问题')
        print('适用于动态规划求解的最优化问题必须具有的第二个要素是子问题的空间要很小，也就是用来解原问题的递归算法可反复地解同样的子问题，而不是总在产生新的问题')
        print('典型地，不同的子问题数是输入规模的一个多项式。当一个递归算法不断地调用同一问题时，我们说该最优问题包含重叠子问题')
        print('相反地，适合用分治法解决的问题往往在递归的每一步都产生全新的问题。',
            '动态规划算法总是充分利用重叠子问题，即通过每个子问题只解一次，把解保存在一个在需要时就可以查看的表中，而每次查表的时间为常数')
        print('动态规划要求其子问题即要独立又要重叠')
        # !动态规划最好存储子问题的结果在表格中，省时省力
        print('做备忘录')
        # !备忘录动态规划填表时更像递归版本，即动态规划的递归版本
        print('动态规划有一种变形，它既具有通常的动态规划方法的效率，又采用了一种自顶向下的策略。其思想就是备忘原问题的自然但是低效的递归算法')
        print('像在通常的动态规划中一样，维护一个记录了子问题解的表，但有关填表动作的控制结构更像递归算法')
        print('加了备忘录的递归算法为每一个子问题的解在表中记录一个表项。开始时，每个表项最初都包含一个特殊的值，以表示该表项有待填入')
        print('总之，矩阵链乘法问题可以在O(n^3)时间内，用自顶向下的备忘录算法或自底向上的动态规划算法解决')
        print('两种方法都利用了重叠子问题的性质。原问题共有Θ(n^2)个不同的子问题，这两种方法对每个子问题都只计算一次。',
            '如果不使用做备忘录,则自然递归算法就要以指数时间运行，因为它要反复解已经解过的子问题')
        print('在实际应用中，如果所有的子问题都至少要被计算一次，则一个自底向上的动态规划算法通常要比一个自顶向下的做备忘录算法好出一个常数因子，',
            '因为前者无需递归的代价，而且维护表格的开销也小一点')
        print('此外，在有些问题中，还可以用动态规划算法中的表存取模式来进一步减少时间或空间上的需求')
        print('练习15.3-1 RECURSIVE-MATRIX-CHAIN要比枚举对乘积所有可能的加全部括号并逐一计算其乘法的次数')
        print('练习15.3-2 请解释在加速一个好的分治算法如合并排序方面，做备忘录方法为什么没有效果。',
            '因为分治算法子问题并没有重复和最优，只是一个解的过程。合并排序谁与谁合并已经确定')
        print('练习15.3-3 考虑矩阵链乘法问题的一个变形，其目标是加全部括号矩阵序列以最大化而不是最小化标量乘法的次数。这个问题具有最优子结构')
        print('练习15.3-4 描述装配线调度问题如何具有重叠子问题')
        print('练习15.3-5 在动态规划中，我们先求解各个子问题，我们先求解各个子问题，然后再来决定该选择它们中的哪一个来用在原问题的最优解中。')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_4:
    '''
    chpater15.4 note and function
    '''
    def lcs_length(self, x, y):
        '''
        计算LCS的长度(也是矩阵路径的解法) 时间复杂度`O(mn)`

        Return
        ===
        (b ,c)
        '''
        m = len(x)
        n = len(y)
        c = zeros([m + 1, n + 1])
        b = zeros((m, n), dtype=np.str)
        for i in range(0, m):
            for j in range(0, n):
                if x[i] == y[j]:
                    c[i + 1][j + 1] = c[i][j] + 1
                    b[i][j] = '↖'
                elif c[i][j + 1] >= c[i + 1][j]:
                    c[i + 1][j + 1] = c[i][j + 1]
                    b[i][j] = '↑'
                else:
                    c[i + 1][j + 1] = c[i + 1][j]
                    b[i][j] = '←'
        return (c, b)

    def __lookup_lcs_length(self, x, y, c, b, i, j):
        if c[i][j] != math.inf:
            return c[i][j]
        if x[i - 1] == y[j - 1]:
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i - 1, j - 1) + 1
            b[i - 1][j - 1] = '↖'
        elif self.__lookup_lcs_length(x, y, c, b, i - 1, j) >= \
            self.__lookup_lcs_length(x, y, c, b, i, j - 1):
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i - 1, j)
            b[i - 1][j - 1] = '↑'
        else:
            c[i][j] = self.__lookup_lcs_length(x, y, c, b, i, j - 1)
            b[i - 1][j -1] = '←'
        return c[i][j]

    def memoized_lcs_length(self, x, y):
        '''
        公共子序列的备忘录版本 时间复杂度`O(mn)`
        '''
        m = len(x)
        n = len(y)
        c = zeros([m + 1, n + 1])
        b = zeros((m, n), dtype=np.str)
        #b = '↓'
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c[i][j] = math.inf
        for i in range(0, m):
            for j in range(0, n):
                b[i][j] = '↓'
        self.__lookup_lcs_length(x, y, c, b, m, n)
        return (c, b)

    def memoized_lcs_show(self, x, y):
        '''
        公共子序列的备忘录版本打印公共子序列 时间复杂度`O(mn)`
        '''
        c, b = self.memoized_lcs_length(x, y)
        print(c)
        print(b)
        self.print_lcs(b, x, len(x) - 1, len(y) - 1)
        print('')

    def print_lcs(self, b, X, i, j):
        '''
        打印公共子序列 运行时间为`O(m + n)`
        '''
        if i == -1 or j == -1:
            return
        if b[i ,j] == '↖':
            self.print_lcs(b, X, i - 1, j - 1)
            print(X[i], end=' ')
        elif b[i, j] == '↑':
            self.print_lcs(b, X, i - 1, j)
        else:
            self.print_lcs(b, X, i, j - 1)

    def print_lcs_with_tablec(self, c, X, Y, i, j):
        '''
        打印公共子序列 运行时间为`O(m + n)`
        '''
        if i == -2 or j == -2:
            return
        if c[i ,j] == c[i - 1][j - 1] + 1 and X[i] == Y[j]:
            self.print_lcs_with_tablec(c, X, Y, i - 1, j - 1)
            print(X[i], end=' ')
        elif c[i - 1, j] >= c[i][j - 1]:
            self.print_lcs_with_tablec(c, X, Y, i - 1, j)
        else:
            self.print_lcs_with_tablec(c, X, Y, i, j - 1)

    def longest_inc_seq(self, x):
        '''
        最长递增子序列(动态规划求解) `O(n^2)` 

        Example
        ===
        ```python
        >>> longest_inc_seq([2, 3, 1, 4])
        >>> [2, 3, 4]
        ```
        '''
        # 序列的长度
        n = len(x)
        # 动态规划子问题表的深度
        t = zeros([n, n])
        for i in range(n):
            for j in range(n):
                t[i][j] = math.inf
        last = 0
        max_count = 0
        max_count_index = 0
        seq = []
        for i in range(n):
            top = 0
            count = 1
            for j in range(i, n):
                if x[i] <= x[j] and top <= x[j]:
                    t[i][j] = x[j]
                    count += 1
                    top = x[j]
                    if count >= max_count:
                        max_count = count
                        max_count_index = i
                else:
                    t[i][j] = math.inf
        for i in range(n):
            val = t[max_count_index][i]
            if val != math.inf:
                seq.append(val)
        print(t)
        return seq

    def lower_bound(self, arr, x, start, end):
        '''
        二分查找数组`arr`中大于`x`的元素的最小值
        '''
        middle = (start + end) // 2
        while arr[middle] < x:
            middle -= 1
        return middle

    def fast_longest_inc_seq(self, x):
        '''
        快速递归的最长递增子序列(二分查找) `O(nlgn)`
        '''
        n = len(x)
        g = []
        l = []
        # O(n)
        for i in range(n):
            g.append(math.inf)
        for i in range(n):
            # 二分查找 O(nlgn)
            k = self.lower_bound(g, x[i], 0, n -1)
            g[k] = x[i]
        # quick sort O(nlgn)
        g.sort()
        for i in range(n):
            if g[i] != math.inf:
                l.append(g[i])
        return l

    def note(self):
        '''
        Summary
        ====
        Print chapter15.4 note

        Example
        ====
        ```python
        Chapter15_4().note()
        ```
        '''
        print('chapter15.4 note as follow')   
        print('15.4 最长公共子序列')
        print('在生物学应用中，经常要比较两个(或更多)不同有机体的DNA。一个DNA螺旋由一串被称为基的分子组成')
        print('可能的基包括腺嘌呤，鸟嘌呤，胞嘧啶，胸腺嘧啶')
        print('分别以它们的首字母来表示这些基，一个DNA螺旋可以表示为在有穷集合{A,C,G,T}上的一个串')
        print('如一个有机体的DNA串可能为S1=ACCGTACGAT,而另一个有机体的DNA可能为S2=GTCCTTCGAT')
        print('将两个DNA螺旋作比较的一个目的就是要确定这两个螺旋有多么相似')
        print('目的是找出第三个螺旋S3,在S3中的基也都出现在S1和S2中；而且这些基必须是以相同的顺序出现，但是不必要是连续的')
        print('能找到的S3越长，S1和S2就越相似')
        print('将这个相似度概念形式化为最长公共子序列问题。',
            '一个给定序列的子序列就是该给定序列中去掉零个或者多个元素')
        print('例如，Z=<B,C,D,B>是X=<A,B,C,B,D,A,B>的一个子序列，相应的下标序列为<2,3,5,7>')
        print('如果Z既是X的一个子序列又是Y的一个子序列，称序列Z是X和Y的公共子序列')
        print('例如：X=<A,B,C,B,D,A,B>,Y=<B,D,C,A,B,A>则序列<B,C,A>即为X和Y的一个公共子序列')
        print('但是<B,C,A>不是X和Y的一个最长公共子序列(LCS),因为它的长度等于3')
        print('而同为X和Y的公共子序列<B,C,B,A>其长度等于4。序列<B,C,B,A>是X和Y的一个LCS')
        print('<B,D,A,B>也是，因为没有长度为5或更大的公共子序列')
        print('LCS问题可用动态规划来有效解决')
        print('步骤1.描述一个最长公共子序列')
        print(' 定理15.1，设X和Y为两个序列，并设Z为X和Y任意一个LCS(最长公共子序列)')
        print(' 1) 如果xm=yn,那么zk=xm=yn,而且Z(k-1)是Xm-1和Yn-1的一个LCS')
        print(' 2) 如果xm≠yn,那么zk≠xm,蕴含Z是Xm-1和Y的一个LCS')
        print(' 3) 如果xm≠yn,那么zk≠yn,蕴含Z是X和Yn-1的一个LCS')
        print('步骤2.一个递归解')
        print(' 寻找LCS时，可能要检查一个或两个子问题。如果xm=yn,必须找出Xm-1和Yn-1的一个LCS')
        print(' 将xm=yn添加到这个LCS上，可以产生X和Y的一个LCS。如果xm≠yn，就必须解决两个子问题：')
        print(' 找出Xm-1和Y的一个LCS，以及找出X和Yn-1的一个LCS。')
        print(' 在这两个LCS，较长的就是X和Y的一个LCS，因为这些情况涉及了所有的可能，其中一个最优的子问题解必须被使用在X和Y的一个LCS中')
        print(' LCS问题的中的重叠子问题，以及共享子子问题')
        print(' 像在矩阵链乘法问题中一样，LCS问题的递归解涉及到建立一个最优解的值的递归式。定义c[i,j]为序列Xi和Yi的一个LCS的长度')
        print(' 递归式子：')
        print('  c[i,j] = 0; 如果i = 0 或 j = 0')
        print('  c[i,j] = c[i-1,j-1]+1; 如果i,j>0和xi=yj')
        print('  c[i,j]=max(c[i,j-1],c[i-1,j]); 如果i,j>0和xi≠yj')
        print('步骤3.计算LCS的长度')
        print(' 容易写出一个指数时间的递归算法，来计算连个序列的LCS的长度，',
            '因为只有Θ(mn)个不同的子问题,所以可以用动态规划来自底向上计算解')
        X = ['A', 'B', 'C', 'B', 'D', 'A', 'B']
        Y = ['B', 'D', 'C', 'A', 'B', 'A']
        c, b = self.lcs_length(X, Y)
        print('the c is')
        print(c)
        print('the b is')
        print(b)
        self.print_lcs(b, X, len(X) - 1, len(Y) - 1)
        print('')
        print('改进代码')
        print('一旦涉及出某个算法之后，常常可以在时间内或空间上对该算法做些改进。对直观的动态规划算法尤为如此')
        print('有些改变可以简化代码并改进一些常数因子，但并不会带来算法性能方面的渐进改善。',
            '其他一些改变则可以可以在时间和空间上有相当大的改善')
        print('其他一些改变则可以在时间和空间上有相当大的渐进节省')
        print('在求公共子序列当中，完全可以去掉b。每个表项c[i, j]仅依赖于另外三个c表项：c[i-1, j-1], c[i-1,j]和c[i,j-1]')
        print('给定c[i, j]的值，我们可在O(1)时间内确定这三个值中的哪一个被用来计算c[i, j]，而不检查表b')
        print('然而，我们能减少LCS-LENGTH的渐进空间需求，因为它一次只需表c的两行：正在被计算的一行和前面一行')
        print('如果仅要求求出一个LCS的长度，则这种改进是有用的；如果要重构一个LCS的元素，',
            '则小的表无法包含足够的信息来使我们在O(m+n)时间内重新执行以前各步')
        print('练习15.4-1 ')
        X = ['1', '0', '0', '1', '0', '1', '0', '1']
        Y = ['0', '1', '0', '1', '1', '0', '1', '1', '0']
        c, b = self.lcs_length(X, Y)
        print('the c is')
        print(c)
        print('the b is')
        print(b)
        self.print_lcs(b, X, len(X) - 1, len(Y) - 1)
        print(' ')
        self.print_lcs_with_tablec(c, X, Y, len(X) - 1, len(Y) - 1)
        print(' ')
        print('练习15.4-2 利用表c中拐点的元素，c矩阵中元素是它斜上方元素+1，且x[i]==y[j]，说明是↖️')
        print('练习15.4-3 请给出一个LCS-LENGTH的运行时间为O(mn)的做备忘录版本')
        self.memoized_lcs_show(X, Y)
        print('练习15.4-4 略')
        print('练习15.4-5 求n个数的序列中最长的单调递增子序列，O(n^2)')
        print(self.longest_inc_seq([1, 3, 5, 7, 1, 2, 3, 4, 5, 7]))
        print(self.longest_inc_seq([5, 4, 3, 7, 1, 2, 3, 6, 2, 8]))
        print(self.longest_inc_seq([1, 2, 3, 4, 5, 2, 3, 1, 9]))
        print('练习15.4-6 求n个数的序列中最长的单调递增子序列，O(nlgn)')
        print(self.fast_longest_inc_seq([1, 3, 5, 7, 1, 2, 3, 4, 5, 7]))
        print(self.fast_longest_inc_seq([5, 4, 3, 7, 1, 2, 3, 6, 2, 8]))
        print(self.fast_longest_inc_seq([1, 2, 3, 4, 5, 2, 3, 1, 9]))
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

class Chapter15_5:
    '''
    chpater15.5 note and function
    '''

    def optimal_bst(self, p, q, n):
        '''
        求最优二叉树
        '''
        e = zeros((n + 2, n + 1))
        w = zeros((n + 2, n + 1))
        root = zeros((n, n))
        for i in range(1, n + 2):
            e[i][i - 1] = q[i - 1]
            w[i][i - 1] = q[i - 1]
        for l in range(1, n + 1):
            for i in range(1, n - l + 1 + 1):
                j = i + l - 1
                e[i][j] = math.inf
                w[i][j] = w[i][j - 1] + p[j] + q[j]
                for r in range(i, j + 1):
                    t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                    if t < e[i][j]:
                        e[i][j] = t
                        root[i - 1][j - 1] = r
        e_return = zeros((n + 1, n + 1))
        w_return = zeros((n + 1, n + 1))
        for i in range(n):
            e_return[i] = e[i + 1]
            w_return[i] = w[i + 1]
        return (e_return, root)
    
    def construct_optimal_bst(self, root):
        '''
        给定表root，输出一棵最优二叉查找树的结构
        '''
        i = 0
        j = 0
        count = shape(root)[-1]

    def __compute_weight(self, i : int, j : int, key : list, fkey : list, weight):
        if i - 1 == j:
            weight[i][j] = fkey[j]
        else:
            weight[i][j] = self.__compute_weight(i, j - 1, key, fkey, weight) + key[j] + fkey[j]
        return weight[i][j]
            
    def __dealbestBSTree(self, i : int, j : int, key : list, fkey : list, weight, min_weight_arr):
        '''
        备忘录模式(从上到下模式)
        '''
        if i - 1 == j:
            min_weight_arr[i][j] = weight[i][j]
            return weight[i][j]
        if min_weight_arr[i][j] != 0:
            return min_weight_arr[i][j]
        _min = 10
        for k in range(i, j + 1):
            tmp = self.__dealbestBSTree(i, k - 1, key, fkey, weight, min_weight_arr) + \
                self.__dealbestBSTree(k + 1, j, key, fkey, weight, min_weight_arr) + \
                weight[i][j]
            if tmp < _min:
                _min = tmp
        min_weight_arr[i][j] = _min
        return _min

    def bestBSTree(self, key : list, fkey : list):
        '''
        最优二叉搜索树的算法实现，这里首先采用自上而下的求解方法(动态规划+递归实现) `O(n^3)`
        '''
        n = len(key)
        min_weight_arr = zeros((n + 1, n))
        weight = zeros((n + 1, n))
        for k in range(1, n + 1):
            self.__compute_weight(k, n - 1, key, fkey, weight)
        self.__dealbestBSTree(1, n - 1, key, fkey, weight, min_weight_arr)
        m_w_r = zeros((n, n))
        w_r = zeros((n, n))
        for i in range(n):
            m_w_r[i] = min_weight_arr[i + 1]
            w_r[i] = weight[i + 1]
        return (w_r, m_w_r, min_weight_arr[1][n - 1]) 

    def show_bestBSTree(self, key : list, fkey : list):
        '''
        最优二叉搜索树的算法实现，这里首先采用自上而下的求解方法(动态规划+递归实现) `O(n^3)`
        并且打印出权重矩阵和最小权重
        '''
        w, m, min = self.bestBSTree(key, fkey)
        print('the weight matrix is')
        print(w)
        print('the min weight matrix is')
        print(m)
        print('the min weight value is')
        print(min)

    def note(self):
        '''
        Summary
        ====
        Print chapter15.5 note

        Example
        ====
        ```python
        Chapter15_5().note()
        ```
        '''
        print('chapter15.5 note as follow')   
        print('15.5 最优二叉查找树')
        # !使用动态规划求解最优二叉查找树
        print('如在一篇文章中搜索单词，希望所花费的总时间尽量地小，',
            '可以使用红黑树或者任何其他的平衡二叉查找树来保证每个单词O(lgn)的搜索时间')
        print('但是每个单词出现的频率并不同，而且在二叉查找树中搜索一个关键字时，访问的结点个数等于1加上包含该关键字的结点的深度')
        print('假设知道每个单词出现的频率，应该如何组织一棵二叉查找树，使得所有的搜索访问的结点数目最小呢？')
        print('最优二叉查找树。形式地：给定一个由n个互异的关键字组成的序列K=<k1,k2,...,kn>,且关键字有序<k1<k2<...<kn>')
        print('对每个关键字ki,一次搜索为ki的概率是pi。某些搜索的值可能不在K内，',
            '因此有n+1个"虚拟键"d0,d1,d2,...,dn代表不在K内的值。',
            '具体地，d0代表所有小于k1的值,dn代表所有大于kn的值',
            '而对于i=1,2,...,n-1,虚拟键di代表所有位于ki和ki+1之间的值')
        print('因为已知了每个关键字和每个虚拟键被搜索的概率，因而可以确定一棵给定的二叉查找树T内一次搜索的期望代价')
        print('对给定一组概率，目标是构造一个期望搜索代价最小的二叉查找树。把这种树称作最优二叉查找树')
        print('一棵最优二叉查找树不一定是一棵整体高度最小的树。也不一定总是把有最大概率的关键字放在根部来构造一棵最优二叉查找树')
        print('如同矩阵链乘法，穷举地检查所有的可能性不会得到一个有效的算法，可以将任何n个结点的二叉树的结点以关键字k1,k2,...kn来标识')
        print('构造一棵最优二叉查找树，然后添加虚拟键作叶子。看到n个结点的二叉树共有Ω(4^n/n^(3/2))个，',
            '所以在一个穷举搜索中，必须检查指数个数的二叉查找树。使用动态规划解这个问题')
        print('步骤1.一棵最优二叉查找树的结构')
        print(' 最优子结构：如果一棵最优二叉查找树T有一棵包含关键字ki,...,kj的子树T1,',
            '那么这颗子树T1对于关键字ki,...kj和虚拟键di-1,...,dj的子问题也必定是最优的')
        print(' 如果有一棵子树T2,其期望代价比T1小，那么可以把T1从T中剪下，然后贴上T2，而产生一个期望代价比T小的二叉查找树，这与T的最优性相矛盾')
        print(' 使用最优子结构来说明可以根据子问题的最优解,来构造原问题的一个最优解')
        print(' 约定：这些子树同时也包含虚拟键，即一棵包含关键字的子树没有真实的关键字但包含单一的虚拟键di-1')
        print('步骤2.一个递归解')
        print(' 选取子问题域为找一个包含关键字ki,...,kj的最优二叉查找树，其中i>=1而且j>=i-1。')
        print(' 定义e[i, j]为搜索一棵包含关键字ki,...,kj的最优二叉查找树的期望代价。最终需要计算e[1, n]')
        print(' 当j=i-1时出现简单情况。此时只有虚拟键di-1。期望的搜索代价是e[i, i-1]=qi-1')
        print(' 结论:选择有最低期望搜索代价的结点作为根，从而得到最终的递归公式：')
        print(' e[i, j]=qi-1 如果 j = i - 1')
        print(' e[i ,j]=min{e[i,r-1]+e[r+1,j]+w(i,j)} 如果i<=j')
        print(' e[i, j]的值是在最优二叉查找树中的期望搜索代价。为有助于记录最优二叉查找树的结构，定义root[i, j]为kr的下标r')
        print('步骤3：计算一棵最优二叉查找树的期望搜索代价')
        print(' 最优二叉查找树与矩阵链乘法的特征之间有一些相似。在二者的问题域中，子问题由连续的下标范围组成')
        print(' 直接递归式的实现和直接递归的矩阵链乘法一样低效')
        print(' 为了提高效率，还需要一个表格。不是每当计算e[i, j]时都从头开始计算w(i, j),',
            '而是把这些值保存在表w[1..n+1,0..n]中')
        print(' 因此，可以计算出Θ(n^2)个w[i, j]的值，每一个值需要Θ(1)的计算时间')
        print('OPTIMAL-BST计算出的表e[i][j],w[i][j]和root[i][j]')
        print('OPTIMAL-BST过程需要Θ(n^3)的运行时间，这与MATRIX-CHAIN-ORDER是一样的，',
            '因为for循环有三层嵌套,而且每个循环的下标有至多n个值')
        print('练习15.5-1 写出过程CONSTRUCT-OPTIMAL-BST(root)的伪代码，给定表root，输出一棵最优二叉查找树的结构')
        p = [0, 0.15, 0.10, 0.05, 0.10, 0.20]
        q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]
        e, root = self.optimal_bst(p, q, len(q) - 1)
        print(e)
        print(root)
        self.show_bestBSTree(p, q)
        self.construct_optimal_bst(root)
        print('练习15.5-2 对n=7个关键字以及如下概率的集合，确定一棵最优二叉查找树的代价和结构')
        # p的第一个元素是用不到的，k的下标从1开始
        p = [0, 0.04, 0.06, 0.08, 0.02, 0.10, 0.12, 0.14]
        q = [0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05]
        e, root = self.optimal_bst(p, q, len(q) - 1)
        print(e)
        print(root)
        self.show_bestBSTree(p, q)
        self.construct_optimal_bst(root)
        print('练习15.5-3 略')
        print('练习15.5-4 ')
        print('思考题15-1 双调欧几里得旅行商问题')
        print('思考题15-2 整齐打印')
        print('思考题15-3 编辑距离')
        print('思考题15-4 计划一个公司聚会')
        print('思考题15-5 Viterbi算法')
        print('思考题15-6 在棋盘上移动')
        print('思考题15-7 达到最高效益的调度')
        # python src/chapter15/chapter15note.py
        # python3 src/chapter15/chapter15note.py

chapter15_1 = Chapter15_1()
chapter15_2 = Chapter15_2()
chapter15_3 = Chapter15_3()
chapter15_4 = Chapter15_4()
chapter15_5 = Chapter15_5()

def printchapter15note():
    '''
    print chapter15 note.
    '''
    print('Run main : single chapter fiveteen!')  
    chapter15_1.note()
    chapter15_2.note()
    chapter15_3.note()
    chapter15_4.note()
    chapter15_5.note()

# python src/chapter15/chapter15note.py
# python3 src/chapter15/chapter15note.py
if __name__ == '__main__':  
    printchapter15note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter16/chapter16note.py
# python3 src/chapter16/chapter16note.py
'''

Class Chapter16_1

Class Chapter16_2

Class Chapter16_3

Class Chpater16_4

Class Chapter16_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import *

if __name__ == '__main__':
    import huffmantree as ht
else:
    from . import huffmantree as ht

class Chapter16_1:
    '''
    chpater16.1 note and function
    '''
    def recursive_activity_selector(self, s, f, i, j):
        '''
        递归解决活动选择问题
        '''
        m = i + 1
        while m < j and s[m] < f[i]:
            m = m + 1
        if m < j:
            return [m] + self.recursive_activity_selector(s, f, m, j) 
        else:
            return []

    def greedy_activity_selector(self, s, f):
        '''
        迭代贪心算法解决活动选择问题
        '''
        n = len(s)
        A = [1]
        i = 1
        for m in range(2, n):
            if s[m] >= f[i]:
                A = A + [m]
                i = m
        return A

    def normal_activity_selector(self, s, f):
        '''
        常规for循环解决选择问题
        '''
        A = []
        n = len(s)
        c = np.zeros((n, n))
        length = np.zeros(n)
        for k in range(n):
            start = s[k]
            end = f[k]
            c[k][k] = k + 1
            length[k] += (end - start)
            for i in range(k):
                if f[i] < start:
                    start = s[i]
                    c[k][i] = i + 1
                    length[k] += (f[i] - s[i])
            for j in range(k + 1, n):
                if s[j] >= end:
                    end = f[j]
                    c[k][j] = j + 1
                    length[k] += (f[j] - s[j])
        return c, length

    def dp_activity_selector(self, s, f):
        '''
        动态规划解决选择问题
        '''
        n = len(s)
        c = np.zeros((n, n))
        index = np.zeros((n, n))
        for step in range(2, n):
            for i in range(0, n - 1):
                j = step + i
                if j < n:
                    if f[i] <= s[j]:
                        for k in range(i + 1, j):
                            if f[k] > s[j] or s[k] < f[i]:
                                continue
                            result = c[i][k] + c[k][j] + 1
                            if result > c[i][j]:
                                c[i][j] = result
                                index[i][j] = k
        return index

    def dp_activity_selector_print(self, index, i, j):
        '''
        打印结果
        '''
        k = int(index[i][j])
        if k != 0:
            self.dp_activity_selector_print(index, i, k)
            print(k, end=' ')
            self.dp_activity_selector_print(index, k, j)
        
    def note(self):
        '''
        Summary
        ====
        Print chapter16.1 note

        Example
        ====
        ```python
        Chapter16_1().note()
        ```
        '''
        print('chapter16.1 note as follow')   
        print('第16章 贪心算法')
        # !适用于最优化问题的算法往往具有一系列步骤，每一步有一组选择
        print('适用于最优化问题的算法往往具有一系列步骤，每一步有一组选择')
        print('对许多最优化的问题来说，采用动态规划有点大材小用的意思，只要采用另一些更简单的方法就行了')
        # !贪心算法使得所做的选择在当前看起来都是最佳的，期望通过所做的局部最优得到最终的全局最优
        print('贪心算法使得所做的选择在当前看起来都是最佳的，期望通过所做的局部最优得到最终的全局最优')
        print('贪心算法对大多数最优化问题都能产生最优解，但也不一定总是这样的')
        print('在讨论贪心算法前，首先讨论动态规划方法，然后证明总能用贪心的选择得到其最优解')
        print('16.2给出一种证明贪心算法正确的方法')
        # !有许多被视为贪心算法应用的算法，如最小生成树，Dijkstra的单源最短路径，贪心集合覆盖启发式
        print('有许多被视为贪心算法应用的算法，如最小生成树，Dijkstra的单源最短路径，贪心集合覆盖启发式')
        print('16.1 活动选择问题')
        # !活动选择问题：对几个互相竞争的活动进行调度,它们都要求以独占的方式使用某一公共资源
        print('对几个互相竞争的活动进行调度,它们都要求以独占的方式使用某一公共资源')
        print('设有n个活动和某一个单独资源构成的集合S={a1,a2,...,an}，该资源一次只能被一个活动占用')
        print('各活动已经按照结束时间的递增进行了排序')
        print('每个活动ai都有一个开始时间si和一个结束时间fi，资源一旦被活动ai选中后')
        print('活动ai就开始占据左闭右开时间区间，如果两个活动ai，aj的时间没有交集，称ai和aj是兼容的')
        # !活动选择问题就是要选择出一个由互相兼容的问题组成的最大子集合
        print('动态规划方法解决活动选择问题时，将原方法分为两个子问题，',
            '然后将两个子问题的最优解合并成原问题的最优解')
        # !贪心算法只需考虑一个选择(贪心的选择)，再做贪心选择时，子问题之一必然是空的，因此只留下一个非空子问题
        print('贪心算法只需考虑一个选择(贪心的选择)，再做贪心选择时，子问题之一必然是空的，因此只留下一个非空子问题')
        print('因此找到一种递归贪心算法解决活动选择问题')
        print('活动选择问题的最优子结构')
        print(' 首先为活动选择问题找到一个动态规划解，找到问题的最优子结构。')
        print(' 然后利用这一结构，根据子问题的最优解来构造出原问题的一个最优解')
        print(' 定义一个合适的子问题空间Sij是S中活动的子集，其中，每个活动都在活动ai结束之后开始，且在活动aj开始之前结束')
        print(' 实际上，Sij包含了所有与ai和aj兼容的活动，并且与不迟于ai结束和不早于aj开始的活动兼容')
        print(' 假设活动已按照结束时间的单调递增顺序排序：')
        print(' 一个非空子问题Sij的任意解中必包含了某项活动ak，而Sij的任一最优解中都包含了其子问题实例Sik和Skj的最优解。')
        print(' 因此，可以可以构造出Sij中的最大兼容活动子集。',
            '将问题分成两个子问题(找出Sik和Skj的最大兼容活动子集)，找出这些子问题的最大兼容活动子集Aik和Akj',
            '而后形成最大兼容活动子集Aij如:Aij=Aik ∪ {ak} ∪ Akj')
        print(' 整个问题的一个最优解也是S0,n+1的一个解')
        print('一个递归解')
        print(' 动态规划方法的第二步是递归地定义最优解的值。对于活动选择问题')
        print(' 考虑一个非空子集Sij,如果ak在Sij的最大兼容子集中被使用，则子问题Sik和Skj的最大兼容子集也被使用')
        print(' 递归式：c[i, j] = max{c[i, k] + c[k, j] + 1} 如果Sij不是空集')
        print(' 递归式：c[i, j] = 0 如果Sij是空集')
        print('将动态规划解转化为贪心解')
        print(' 到此为止，写出一个表格化的、自底向上的、基于递归式的动态规划算法是不难的')
        i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]
        s = [0, 1, 3, 0 ,5, 3, 5, 6, 8, 8, 2, 12]
        f = [0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]
        print(' 定理16.1 对于任意非空子问题Sij,设am是Sij中具有最早结束时间的活动:fm=min{fk:ak属于Sij}')
        print(' 那么(1) 活动am在Sij的某最大兼容活动子集中被使用')
        print(' (2)子问题Sim为空，所以选择am将使子问题Smj为唯一可能非空的子问题')
        print('递归贪心算法')
        print(' 介绍一种纯贪心的，自顶向下(递归)的算法解决活动选择问题,',
            '假设n个输入活动已经按照结束时间的单调递增顺序排序。否则可以在O(nlgn)时间内将它们以此排序')
        print(self.recursive_activity_selector(s, f, 0, len(s)))
        print(self.greedy_activity_selector(s, f))
        s = [1, 2, 0 ,5, 3, 5, 6, 8, 8, 2, 12]
        f = [4, 5, 6, 7, 8, 9, 10,11,12,13,14]
        print(self.normal_activity_selector(s, f))
        print('练习16.1-1 活动选择问题的动态规划算法')
        s = [0, 1, 3, 0 ,5, 3, 5, 6, 8, 8, 2, 12, _math.inf]
        f = [0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14, _math.inf]
        index = self.dp_activity_selector(s, f)
        print(index)
        self.dp_activity_selector_print(index, 0, len(s) - 1)
        print('')
        print('练习16.1-2 略')
        print('练习16.1-3 区间图着色问题：可作出一个区间图，其顶点为已知的活动，其边连接着不兼容的活动',
            '其边连接着不兼容的活动。为使任两个相邻结点的颜色均不相同，',
            '所需的最少颜色数对应于找出调度给定的所有活动所需的最小教室数')
        print('练习16.1-4 并不是任何用来解决活动选择问题的贪心算法都能给出兼容活动的最大集合')
        print(' 请给出一个例子，说明那种在与已选出的活动兼容的活动中选择生存期最短的方法是行不通的')
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

class Chapter16_2:
    '''
    chpater16.2 note and function
    '''
    def zero_one_knapsack_problem_dp(self, total_weight, item_weight, item_value):
        '''
        0-1背包问题的动态规划方法

        时间复杂度`O(n × total_weight)`

        空间复杂度`O(n × total_weight)`

        Args
        ===
        `total_weight` : 背包能容纳的物品总重量

        `item_weight` : `list` 各物品的重量

        `item_value` : `list` 各物品的价值

        Return
        ===
        `item_index` : 存放入背包的物品索引，一个物品只能存放一次

        Example
        ===
        ```python
        >>> total_weight = 50
        >>> item_weight = [10, 20, 30]
        >>> item_value = [60, 100, 120]
        >>> zero_one_knapsack_problem_dp(total_weight, item_weight, item_value):
        >>> [1, 2]
        ```
        '''
        # 动态规划第一步先选取最优子问题结构,并确立表格
        n = len(item_value) 
        W = total_weight
        V = zeros((n, W + 1))
        w = item_weight
        v = item_value   
        for i in range(1, n):
            for j in range(1, W + 1):
                if j < w[i]:
                    V[i][j] = V[i - 1][j]
                else:
                    V[i][j] = max(V[i - 1][j], V[i - 1][j - w[i]] + v[i])
        item = []
        self.__find_zero_one_knapsack_problem_dp_result(V, w, v, n - 1, W, item)
        return item
    
    def __find_zero_one_knapsack_problem_dp_result(self, V, w, v, i, j, item):
        if i >= 0:
            if V[i][j] == V[i - 1][j]:
                self.__find_zero_one_knapsack_problem_dp_result(V, w, v, i - 1, j, item)
            elif j - w[i] >= 0 and V[i][j] == V[i - 1][j - w[i]] + v[i]:
                item.append(i)
                self.__find_zero_one_knapsack_problem_dp_result(V, w, v, i - 1, j - w[i], item)

    def partof_knapsack_problem_ga(self, total_weight, item_weight, item_value):
        '''
        部分背包问题的贪心算法

        Args
        ===
        `total_weight` : 背包能容纳的物品总重量

        `item_weight` : `list` 各物品的重量

        `item_value` : `list` 各物品的价值

        Return
        ===
        `item_index` : 存放入背包的物品索引，一个物品只能存放一次

        Example
        ===
        ```python
        >>> total_weight = 50
        >>> item_weight = [10, 20, 30]
        >>> item_value = [60, 100, 120]
        >>> partof_knapsack_problem_ga(total_weight, item_weight, item_value):
        ```
        '''
        w = item_weight
        v = item_value
        n = len(w)
        r = []
        m = total_weight
        for i in range(n):
            r.append(v[i] * 1.0 / w[i])
        # 冒泡排序
        for i in range(1, n):
            for j in range(n - i):
                # 排序
                if r[j] < r[j + 1]:
                    r[j], r[j + 1] = r[j + 1], r[j]
                    w[j], w[j + 1] = w[j + 1], w[j]
                    v[j], v[j + 1] = v[j + 1], v[j]    
        i = 0 
        while m > 0:
            if w[i] <= m:
                m -= w[i]
                print('value:{} weight:{}'.format(v[i], w[i]))
                i += 1
            else:
                print('value:{} weight:{}'.format(v[i], m))
                m = 0

    def cal_compose_value(self, A, B):
        '''
        计算组合价值
        '''
        assert len(A) == len(B)
        n = len(A)
        value = 0
        for i in range(n):
            value += A[i] ** B[i]
        return value

    def insertsort(self, array, start ,end, isAscending=True):
        '''
        Summary
        ===
        插入排序的升序排列(带排序索引), 原地排序
        
        Parameter
        ===
        `array` : a list like

        `start` : sort start index

        `end` : sort end index

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> array = [6, 5, 4, 3, 2, 1]
        >>> Chapter2_3().insert(array, 1, 4)
        >>> [6 ,2, 3, 4, 5, 1]
        ```
        '''
        if isAscending == True:
            A = array
            for j in range(start + 1, end + 1):
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
        else:
            A = array
            for j in range(start + 1, end + 1):
                ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
                # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
                # 后来摸的牌
                key = A[j]
                # 之前手中的已经排序好的牌的最大索引
                i = j - 1
                # 开始寻找插入的位置并且移动牌
                while(i >= 0 and A[i] <= key):
                    # 向右移动牌
                    A[i + 1] = A[i]
                    # 遍历之前的牌
                    i = i - 1
                # 后来摸的牌插入相应的位置
                A[i + 1] = key
                # 输出升序排序后的牌
        return A

    def max_compose_value(self, A, B):
        ''' 
        最大化报酬问题，对集合`A` 和 集合`B`排序后，使价值最大 (贪心求解)

        value = argmax(∏ ai ** bi)

        '''
        assert len(A) == len(B)
        n = len(A)
        for i in range(n):
            A = self.insertsort(A, i, n - 1, isAscending=False)
            if A[i] >= 1:
                B = self.insertsort(B, i, n - 1, isAscending=False)  
            else:
                B = self.insertsort(B, i, n - 1, isAscending=True)  
        return self.cal_compose_value(A, B)

    def note(self):
        '''
        Summary
        ====
        Print chapter16.2 note

        Example
        ====
        ```python
        Chapter16_2().note()
        ```
        '''
        print('chapter16.2 note as follow')
        print('16.2 贪心策略的基本内容')
        print('贪心算法是通过做一系列的选择来给出某一问题的最优解。',
            '对算法中的每一个决策点，做一个当时(看起来)是最佳的选择')
        print('这种启发式策略并不是总能产生出最优解，但是常常能给出最优解')
        print('开发一个贪心算法遵循的过程：')
        print(' 1) 决定问题的最优子结构')
        print(' 2) 设计出一个递归解')
        print(' 3) 在递归的任一阶段，最优选择之一总是贪心选择。那么做贪心选择总是安全的')
        print(' 4) 证明通过做贪心选择，所有子问题(除一个以外)都为空')
        print(' 5) 设计出一个实现贪心策略的递归算法')
        print(' 6) 将递归算法转换成迭代算法')
        print('通过这些步骤，可以清楚地发现动态规划是贪心算法的基础。')
        print('实际在设计贪心算法时，经常简化以上步骤，通常直接作出贪心选择来构造子结构，',
            '以产生一个待优化解决的子问题')
        print('无论如何，在每一个贪心算法的下面，总会有一个更加复杂的动态规划解。')
        print('可根据如下的步骤来设计贪心算法：')
        print(' 1) 将优化问题转化成这样的一个问题，即先做出选择，再解决剩下的一个子问题')
        print(' 2) 证明原问题总是有一个最优解是做贪心选择得到的，从而说明贪心选择的安全')
        print(' 3) 说明在做出贪心选择后，剩余的子问题具有这样一个性质。',
            '即如果将子问题的最优解和所做的贪心选择联合起来，可以得出原问题的一个最优解')
        # !贪心算法一般不能够解决一个特定的最优化问题，但是贪心选择的性质和最优子结构时两个关键的特点
        print('贪心算法一般不能够解决一个特定的最优化问题，但是贪心选择的性质和最优子结构时两个关键的特点')
        print('如果能够证明问题具有贪心选择性质和最优子结构，那么就可以设计出它的一个贪心算法')
        print('贪心选择性质：')
        print(' 一个全局最优解可以通过局部最优(贪心)选择来达到。换句话说，当考虑做何选择时，',
            '只考虑对当前问题最佳的选择而不考虑子问题的结果')
        print(' 贪心算法不同于动态规划之处。在动态规划中，每一步都要做出选择，但是这些选择依赖于子问题的解')
        print(' 解动态规划问题一般是自底向上，从小子问题处理至大子问题')
        print(' 在贪心算法中，所做的总是当前看似最佳的选择，然后再解决选择之后所出现的子问题')
        print(' 贪心算法所做的当前选择可能要依赖于已经做出的所有选择，',
            '但不依赖于有待于做出的选择或子问题的解')
        print(' 因此，不像动态规划方法那样自底向上地解决子问题，',
            '贪心策略通常是自顶向下地做的，一个一个地做出贪心选择，不断地将给定的问题实例归约为更小的问题')
        print(' 必须证明在每一步所做的贪心选择最终能产生一个全局最优解，这也是需要技巧的所在')
        print(' 贪心选择性质在面对子问题做出选择时，通常能帮助我们提高效率。例如，在活动选择问题中，',
            '假设已将活动按结束时间的单调递增顺序排序，则每个活动只需检查一次。')
        print(' 通常对数据进行处理或选用合适的数据结构(优先队列)，能够使贪心选择更加快速，因而产生出一个高效的算法')
        print('最优子结构')
        print(' 对一个问题来说，如果它的一个最优解包含了其子问题的最优解，则称该问题具有最优子结构')
        print(' 贪心算法中使用最优子结构时，通常是用更直接的方式。结社在原问题中作了一个贪心选择而得到了一个子问题')
        print(' 真正要做的是证明将此子问题的最优解与所做的贪心选择合并后，的确可以得到原问题的一个最优解')
        print(' 这个方案意味着要对子问题采用归纳法，来证明每个步骤中所做的贪心选择最终会产生出一个最优解')
        print('贪心法与动态规划都利用了最优子结构性质')
        print('0-1背包问题是这样的，有一个贼在偷窃一家商店时发现有n件物品；第i件物品值vi元，重wi磅，此处v和w都是整数')
        print('希望带走的东西越值钱越好，但他的背包至多只能装下W磅的东西,要使价值最高，应该带走哪几样东西')
        print('部分背包问题是在0-1背包问题的基础上可以选择带走物品的一部分')
        # !虽然部分0-1背包问题和部分背包问题特别相似，但是部分背包问题可以用贪心策略来解决，而0-1背包问题却不行
        print('虽然部分0-1背包问题和部分背包问题特别相似，但是部分背包问题可以用贪心策略来解决，而0-1背包问题却不行')
        print('使用贪心算法解决部分背包问题：先对每件物品计算其每磅的价值vi/wi。按照一种贪心策略，',
            '窃贼开始时对具有最大每磅价值的物品尽量多拿一些。如果他拿完了该物品而仍然可以取一些其他物品时，',
            '他就再取具有次大的每磅价值的物品，一直继续下去，直到不能再取为止。这样，通过按每磅价值来对所有物品排序',
            '贪心算法就可以O(nlgn)时间运行。关于部分背包问题具有贪心选择性质的证明')
        print('一个简单的问题可以说明贪心为什么不适用0-1背包问题，背包能承受的最大重量为50磅')
        print('物品1 重10磅 值60元(每磅6元)；物品2 重20磅 值100元(每磅5元)；物品3 重30磅 值120元(每磅4元)')
        print('按照贪心策略(即只关注当前的最优情况)，就要取物品1，然而最优解是一定不能取物品1的')
        print('最优解取的是物品2和物品3，留下物品1.两种包含物品1的可能解都是次优的')
        print('即贪心策略对0-1背包问题不适用')
        print('然而对于部分背包问题，在按照贪心策略先取物品1以后，确实可以产生一个最优解')
        print('在0-1背包问题中不应该取物品1的原因在与这样无法把背包填满，空余的空间就降低了他的货物的有效每磅价值')
        print('在0-1背包问题中，当我们考虑是否要把一件物品加到背包中时，必须对把该问题加进去的子问题的解与不取该物品的子问题的解进行比较')
        print('由这种方式形成的问题导致了许多重叠子问题(这是动态规划的一个特点)，所以，可以用动态规划来解决0-1背包问题')
        print('练习16.2-1 证明部分背包问题具有贪心选择性质')
        print('练习16.2-2 请给出一个解决0-1背包问题的运行时间为O(n W)的动态规划方法，',
            'n为物品件数，W为窃贼可放入他背包物品的最大重量')
        # 一般动态规划在输入数据中填入首项0
        total_weight = 8
        item_weight = [0, 5, 4, 3, 1]
        item_value = [0, 3, 4, 5, 6]
        print(self.zero_one_knapsack_problem_dp(total_weight, item_weight, item_value))
        total_weight = 8
        item_weight = [2, 3, 4, 5]
        item_value = [3, 4, 5, 6]
        print(self.zero_one_knapsack_problem_dp(total_weight, item_weight, item_value))
        print('贪心算法解部分背包问题')
        self.partof_knapsack_problem_ga(total_weight, item_weight, item_value)
        print('练习16.2-3 从价值高重量轻的开始拿，拿到满为止')
        print('练习16.2-4 略,公路加油问题')
        print('练习16.2-5 请描述一个算法，使之对给定的一实数轴上的点集{x1,x2,...,xn},能确定包含所有',
            '给定点的最小单位闭区间闭集合')
        print('练习16.2-6 带权中位数：在O(nlgn)的最坏情况时间内求出n个元素的带权中位数',
            '说明如何在O(n)时间内解决部分背包问题')
        print('练习16.2-7 最大化报酬问题')
        A = [1.1, 0.2, 3, 4, 5]
        B = [4.3, 4, 3, 2, 1]
        print(self.cal_compose_value(A, B))
        print(self.max_compose_value(A, B))
        print(A)
        print(B)
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

class Chapter16_3:
    '''
    chpater16.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter16.3 note

        Example
        ====
        ```python
        Chapter16_3().note()
        ```
        '''
        print('chapter16.3 note as follow')
        print('16.3 赫夫曼编码')
        # !赫夫曼编码是一种被广泛应用而且非常有效的数据压缩技术,根据数据特征，一般可以压缩20%-90%,这里的数据指的是字符串序列
        print('赫夫曼编码是一种被广泛应用而且非常有效的数据压缩技术')
        print('根据数据特征，一般可以压缩20%-90%,这里的数据指的是字符串序列')
        print('赫夫曼贪心算法使用了一张字符出现频度表')
        print('可变长编码要比固定长度编码好的多，其特点是对频度高的字符赋以短编码，而对频度低的字符赋以较长的一些编码')
        print('比如只用一比特0编码a，四个比特1100编码f，101编码b，100编码c，111编码d，1101编码e')
        print('a的频度为45')
        print('b的频度为13')
        print('c的频度为12')
        print('d的频度为16')
        print('e的频度为9')
        print('f的频度为5')
        print('前缀编码')
        print(' 上述考虑的编码当中，没有一个编码是另一个编码的前缀。这样的编码称为前缀编码')
        print(' 定理：由字符编码技术所获得的最优数据研所总可用某种前缀编码来获得，',
            '因此将注意力集中到前缀编码上并不失一般性')
        print(' 在前缀编码中解码也是很方便的。因为没有一个码是其他码的前缀')
        print(' 只要识别出第一个编码，将它翻译成原文字符，再对余下的编码文件重复这个解码过程即可')
        print(' 在上述的a到f编码当中，可将字符串001011101唯一地分析为0·0·101·1101，因而可解码为aabe')
        print(' 解码过程需要有一种关于前缀编码的方便表示，使得初始编码可以很容易地被识别出来')
        print(' 有一种表示方法就是叶子为给定字符的二叉树，在这种二叉树中，将一个字符的编码解释为从根至该字符的路径')
        print(' 0表示转向左子结点，1表示转向右子结点')
        print(' 注意并不是二叉查找树，因为各结点无需以排序次序出现，且内结点也不包含关键字')
        # !文件的一种最优编码总是由一棵满二叉树来表示的，树中的每个非结点都有两个子结点
        print('文件的一种最优编码总是由一棵满二叉树来表示的，树中的每个非结点都有两个子结点')
        print('二叉树中每个叶子结点被标以一个字符及其出现的频度。')
        print('每个内结点标以其子树中所有叶子的额度总和')
        print('固定长度编码不是最优编码，因为表示它的树不是满二叉树：有的编码开始于10，但没有一个开始于11')
        print('给定对用一种前缀编码的二叉树T，很容易计算出编码一个文件所需的位数。')
        print('对字母表C中的每一个字符c，设f(c)表示c在文件中出现的频度，d(c)表示c的叶子在树中的深度。')
        print('注意d(c)也是字符c的编码的长度。这样编码一个文件所需的位数就是')
        print('  B(T)=∑f(c)d(c)')
        print('构造赫夫曼编码')
        # !赫夫曼设计了一个可用来构造一种称为赫夫曼编码的最优前缀编码的贪心算法 
        print('赫夫曼设计了一个可用来构造一种称为赫夫曼编码的最优前缀编码的贪心算法 ')
        print('该算法的正确性要依赖于贪心选择性质和最优子结构')
        c = ['a', 'b', 'c', 'd', 'e', 'f']
        f = [45, 13, 12, 16, 9, 5]
        tree = ht.HuffmanTreeBuilder(c, f).build()
        print('字符为：')
        print(tree.characters)
        print('频度为：')
        print(tree.fs)
        print('编码为：')
        print(tree.codings)
        print('赫夫曼算法的正确性')
        print(' 为了证明贪心算法赫夫曼的正确性，就要证明确定最优前缀编码的问题局哟与贪心选择和最优子结构性质')
        print('引理16.2 设C为一字母表，其中每个字符c具有频度f[c]。',
            '设x和y为C中具有最低频度的两个字符,则存在C的一种最优前缀编码，其中x和y的编码长度相同但最后一位不同')
        print('证明的主要思想是使树T表示任一种最优前缀编码，然后对它进行修改，',
            '使之表示另一种最优前缀编码，使得字符x和y在新树中成为具有最大深度的兄弟叶结点。',
            '如果能做到这一点，则它们的编码就具有相同长度，而仅仅最后一位不同')
        print('下面的引理证明了构造最优前缀编码的问题具有最优子结构性质')
        print('引理16.3 设C为一给定字母表，其中每个字母c属于C都定义有频度f[c].设x和y是C中具有最低频度的两个字母。',
            '并设C`为字母表移去x和y，再加上(新)字符z后的字母表,C`=C-{x,y}+{z},定义f[z]=f[x]+f[y]')
        print('设T`为表示字母表C`上最优前缀编码的任意一棵树。那么，',
            '将T`中的叶子结点z替换成具有x和y孩子的内部结点所得到的树T，表示字母表C上的一个最优前缀编码')
        print('定理16.4 HUFFMAN过程产生一种最优前缀编码')
        print('练习16.3-1 定理：一棵不满的二叉树不可能与一种最优前缀编码对应')
        print('练习16.3-2 对下面的频度集合（基于前8个斐波那契数），其最优的赫夫曼编码是什么')
        c = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        f = [1, 1, 2, 3, 5, 8, 13, 21]
        tree = ht.HuffmanTreeBuilder(c, f).build()
        print('字符为：')
        print(tree.characters)
        print('频度为：')
        print(tree.fs)
        print('编码为：')
        print(tree.codings)
        print('练习16.3-3 定理：对应于某种编码的树的总代价也能通过计算所有内结点的两子结点的频度之和得到')
        print('练习16.3-4 定理：对一字母表的字符按其频度的单调递减顺序排序，则存在一个编码长度单调递增的最优编码')
        print('练习16.3-5 假设有一个字母表C={0,1,...,n-1}上的最优前缀编码',
            '想用尽可能少的位来传输。证明：C上任意一个最优前缀编码都可由2n-1+n[lgn]个位序列来表示')
        print(' 用2n-1位来说明树的结构，通过树的遍历来发现')
        print('练习16.3-6 将赫夫曼编码推广至三进制编码(用0，1，2来编码)，证明它能产生最优编码')
        print(' 每次取三个最小结点构造三叉树分别编码0，1，2即可')
        print('练习16.3-7 假设某一数据文件包含一系列的8位字符，且所有256个字符的频度都差不多',
            '最大字符频度不到最小频度字符的两倍','证明：这种情况下赫夫曼编码的效率与普通的8位固定长度编码就可以')
        print(' 频度差不多的话用赫夫曼编码出来的编码长度所有字符都差不多')
        print('练习16.3-8 定理：没有一种数据压缩方案能对包含随机选择的8位字符的文件作任何压缩')
        print(' 将文件数与可能的编码文件数进行比较')
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

class Chapter16_4:
    '''
    chpater16.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter16.4 note

        Example
        ====
        ```python
        Chapter16_4().note()
        ```
        '''
        print('chapter16.4 note as follow')
        print('16.4 贪心法的理论基础')
        # !关于贪心算法有一种很漂亮的理论，这一种理论在确定贪心方法何时能产生最优解时非常有用,用到了一种称为\"拟阵\"的组合结构
        print('关于贪心算法有一种很漂亮的理论，这一种理论在确定贪心方法何时能产生最优解时非常有用')
        print('用到了一种称为\"拟阵\"的组合结构')
        print('这种理论没有覆盖贪心方法所适用的所有情况，如16.1的活动选择问题和16.3的赫夫曼编码问题')
        # !拟阵是满足一些条件的序对，并且具有贪心选择性质，具有最优子结构性质
        print('拟阵')
        print(' 定理16.5 如果G=(V, E)是一个无向图，则Mg=(Sg,Lg)是个拟阵')
        print(' 定理16.6 某一拟阵中所有最大独立子集的大小都是相同的')
        print('关于加权拟阵的贪心算法')
        print(' 适宜用贪心算法来获得最优解的许多问题，都可以归结为在加权拟阵中，找出一个具有最大权值的独立子集的问题')
        print('引理16.7 (拟阵具有贪心选择性质),假设M是一个具有权函数w的加权拟阵',
            '且S被按权值的单调减顺序排序。设x为S的第一个使x独立的元素，如果x存在，则存在S的一个包含x的最优子集A')
        print('引理16.8 设M为任意一个拟阵。如果x是S的任意元素，是S的独立子集A的一个扩张，那么x也是空集的一个扩张')
        print('推论16.9 设M为任意一个拟阵。如果集合S中元素x不是空集的扩张，那么x也不会是S的任意独立子集A的一个扩张')
        print('引理16.10 (拟阵具有最优子结构性质) 设x为S中被作用于加权拟阵M的Greedy第一个选择了的元素')
        print(' 找一个包含x的具有最大权值的独立子集的问题，可以归约为找出加权拟阵M的一个具有最大权值的独立子集的问题')
        print('定理16.11(拟阵上贪心算法的正确性) 如果M=(S,l)为具有权函数w的加权拟阵，',
            '则调用Greedy(M,w)返回一个最优子集')
        print('练习16.4-1 证明:(S,l)为一个拟阵，其中S为任一有限集合,l为S的所有大小至多为k的子集构成的集合')
        print('练习16.4-2 给定一个m*n的某域(如实数)上的矩阵T，证明(S,l)是个拟阵，其中S为T的所有列构成的集合')
        print('练习16.4-3 证明:如果(S,l`)的最大独立子集是(S,l)的最大独立子集的补集')
        print('练习16.4-4 对于包含了划分的每个块中至多一个成员的集合A，由所有的集合A构成的集合决定了一个拟阵的独立集合')
        print('练习16.4-5 说明在最优解为具有最小权值的最大独立子集的加权拟阵问题中,',
            '如何改造其权值函数，使之称谓一个标准的加权拟阵问题')
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

class Chapter16_5:
    '''
    chpater16.5 note and function
    '''
    def greedy(self, S, l, w):
        '''
        Args
        ===
        `M` : tuple(S, l) 加权拟阵 

        `w` : 相关的正的权函数

        Return
        ===
        `A` : 最优子集

        '''
        A = []
        B = []
        ind = lexsort((S, w))
        for i in ind:
            B.append(S[i])
        for x in B:
            A = A + [x]
        return A
    
    def task_scheduling(self, S, d, w):
        '''
        贪心算法解决任务调度问题

        Args
        ===
        `S` : n个单位时间任务的集合S

        `d` : 任务的截止时间d,每个任务都是单位时

        `w` : 任务的误时惩罚w

        Return
        ===
        `done` : 最优任务调度
        '''      
        n = len(S)
        done = zeros(n)    
        sum = 0
        # 按照截止时间进行冒泡排序
        for i in range(n - 1):
            for j in range(n - i - 1):
                if w[j] < w[j + 1]:
                    w[j], w[j + 1] = w[j + 1], w[j]
                    d[j], d[j + 1] = d[j + 1], d[j]
        # 求解最优任务调度的任务执行顺序
        for i in range(n):
            for j in range(d[j] + 1):
                k = d[j] - j - 1
                if done[k] == 0:
                    done[k] = 1
                    break
                if k == 0:
                    sum += w[i]
        return done, sum
        
    def note(self):
        '''
        Summary
        ====
        Print chapter16.5 note

        Example
        ====
        ```python
        Chapter16_5().note()
        ```
        '''
        print('chapter16.5 note as follow')
        print('16.5 一个任务调度问题')
        print('有一个可用拟阵来解决的有趣问题，即在单个处理器上对若干个单位时间任务进行最优调度',
            '其中每个任务都有一个截止期限和超时惩罚。这个问题看起来很复杂，但是用贪心算法解决则非常简单')
        print('单位时间任务是一个作业，恰好需要一个单位的时间来运行。给定一个有穷单位时间任务的集合S')
        print('对S的一个调度即为S的一个排列，它规定了各任务的执行顺序。',
            '该调度中的第一个任务开始于时间0，结束于时间1;第二个任务开始于时间1，结束于时间2')
        print('目的是找出S的一个调度，使之最小化因误期而导致的总惩罚')
        print('定理16.13 如果S是一个带期限的单位时间任务的集合，',
            '且l为所有独立的任务集构成的集合，则对应的系统(S,l)是一个拟阵')
        print(' 证明：一个独立的任务子集肯定是独立的')
        print('根据定理16.11 可用一个贪心算法来找出一个具有最大权值的独立的任务集A')
        print(' 然后，可以设计出一个以A中的任务作为其早任务的最优调度。')
        print(' 这种方法对在单一处理器上调度具有期限和惩罚的单位时间任务来说是很有效的。')
        print(' 采用了Greedy后，这个算法的运行时间为O(n^2),因为算法中O(n)次独立性检查的每一次都要花O(n)的时间')
        print('贪心算法解决最优任务调度问题：最优子结构和贪心决策使子问题局部最优')
        # n个单位时间任务的集合S
        S = [1, 2, 3, 4, 5, 6, 7]
        # 任务的截止时间d
        d = [4, 2, 4, 3, 1, 4, 6]
        # 任务的误时惩罚w
        w = [70, 60, 50, 40, 30, 20, 10]
        print(self.task_scheduling(S, d, w))
        w = []
        print('练习16.5-1 调度问题的实例，但要将每个惩罚wi替换成80-wi')
        # n个单位时间任务的集合S
        S = [1, 2, 3, 4, 5, 6, 7]
        # 任务的截止时间d,每个任务都是单位时
        d = [4, 2, 4, 3, 1, 4, 6]
        # 任务的误时惩罚w
        w = [10, 20, 30, 40, 50, 60, 70]
        print(self.task_scheduling(S, d, w))
        print('练习16.5-2 如何利用引理16.12的性质2在O(|A|)时间内，确定一个给定的任务集A是否是独立的')
        print('思考题16-1 找换硬币:考虑用最少的硬币数来找n分钱的问题,假设每个硬币的值都是整数')
        print(' 请给出一个贪心算法，使得所换硬币包括一角、五分的、二角五分的和一分的')
        print(' 先换一角的，再换五分的、依次往下换即可')
        print(' 假设所换硬币的单位是c的幂次方，也就是c,c^1,c^2,...,c^k,c和k均为整数，证明贪心算法总可以产生一个最优解')
        print('思考题16-2 最小化平均结束时间的调度')
        print(' a)假设给定一任务集合S={a1,a2,...,an},其中ai一旦开始，需要pi的单位处理时间来完成。')
        print(' 现在只有一台计算机来运行这些任务，而且计算机上每次只能运行一个任务')
        print(' 设ci为任务ai的结束时间，也就是任务ai处理完成的时间，目标是使平均结束时间最小，即最小化(1/n)∑ ci')
        print(' 使用贪心求解，先使各个任务按照pi从小到大的顺序排序，然后依次执行，使用快速排序，堆排序或者合并排序算法使整个人物调度算法的时间为O(nlgn)')
        print(' b)假设现在并不是所有的任务都可以立即获得。亦即，每个任务再被处理之前都有一个松弛时间ri,同时又允许被抢占')
        print(' 则一个任务后来可被挂起和重新启动')
        print(' 例如一个处理时间为6的任务在时间1开始运行，在时间4被抢占，在时间10恢复，在时间11又被抢占，最后在时间13恢复，在时间15完成。')
        print('  所以这个任务总共运行了6单位的时间，但其运行时间被分为了3段(3,1,2)，则该任务的结束时间为15')
        print(' 仍然使用贪心算法求解，运行时间仍然为没有抢占的调度时间')
        print('思考题16-3 无环子图：略')
        print('思考题16-4 调度问题的变形：略')
        # python src/chapter16/chapter16note.py
        # python3 src/chapter16/chapter16note.py

chapter16_1 = Chapter16_1()
chapter16_2 = Chapter16_2()
chapter16_3 = Chapter16_3()
chapter16_4 = Chapter16_4()
chapter16_5 = Chapter16_5()

def printchapter16note():
    '''
    print chapter16 note.
    '''
    print('Run main : single chapter sixteen!')  
    chapter16_1.note()
    chapter16_2.note()
    chapter16_3.note()
    chapter16_4.note()
    chapter16_5.note()

# python src/chapter16/chapter16note.py
# python3 src/chapter16/chapter16note.py
if __name__ == '__main__':  
    printchapter16note()
else:
    pass

```

```py

from __future__ import division, absolute_import, print_function
from copy import deepcopy as _deepcopy

class HuffmanTreeNode:
    '''
    Huffman二叉树结点
    '''
    def __init__(self, left = None, right = None, f = None, p = None, character=None, index=None):
        '''
        Huffman二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `f` : 结点自身频度

        '''
        self.left = left
        self.right = right
        self.f = f
        self.p = p
        self.character = character
        self.coding = ''
        self.index = None

class HuffmanTree:
    def __init__(self):
        self.root = None
        self.__nodes = []
        self.codings = []
        self.characters = []
        self.fs = []
        self.__coding = ""
        
    def addnode(self, node):
        '''
        加入二叉树结点

        Args
        ===
        `node` : `HuffmanTreeNode` 结点

        '''
        self.__nodes.append(node)

    def buildnodecodingformcharacter(self, node):
        if node is not None:
            if node.p is None:
                return
            if node.p.left == node:
                self.__coding += '0'
            if node.p.right == node:
                self.__coding += '1'
            self.buildnodecodingformcharacter(node.p)
        
    def __findnode(self, f):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if f is None:
            return None
        for node in self.__nodes:
            if f == node.f:
                return node
            if node.left is not None:
                if f == node.left.f:
                    return node.left
            if node.right is not None:
                if f == node.right.f:
                    return node.right
        return None

    def __findnode_f_c(self, f, c):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if f is None:
            return None
        for node in self.__nodes:
            if f == node.f and c == node.character:
                return node
            if node.left is not None:
                if f == node.left.f and c == node.left.character:
                    return node.left
            if node.right is not None:
                if f == node.right.f and c == node.right.character:
                    return node.right
        return None

    def __findnodefromc(self, c):
        '''
        根据`f`从`nodes`中寻找结点
        '''
        if c is None:
            return None
        for node in self.__nodes:
            if c == node.character:
                return node
            if node.left is not None:
                if c == node.left.character:
                    return node.left
            if node.right is not None:
                if c == node.right.character:
                    return node.right
        return None

    def renewall(self):
        '''
        更新/连接/构造二叉树
        '''
        for node in self.__nodes:
            if node.left is not None:
                node.left = self.__findnode_f_c(node.left.f, node.left.character)
                node.left.p = node
            if node.right is not None:
                node.right = self.__findnode_f_c(node.right.f, node.right.character)
                node.right.p = node
    
    def renewnode(self, node):
        '''
        更新/连接/构造二叉树结点
        '''
        if node is None:
            return
        if node.left is not None:
            node.left = self.__findnode_index(node.left.index)
            node.left.p = node
        if node.right is not None:
            node.right = self.__findnode_index(node.right.index)
            node.right.p = node

    def renewallcoding(self, characters):
        n = len(characters)
        for i in range(n):
            c = characters[i]
            node = self.__findnodefromc(c)
            self.__coding = ""
            self.buildnodecodingformcharacter(node)
            if node is not None:
                node.coding = self.__coding[::-1]
                self.codings.append(node.coding)

class HuffmanTreeBuilder:
    '''
    HuffmanTree 构造器
    '''
    def __init__(self, C : list, f : list):
        self.C = C
        self.f = f

    def extract_min(self, C : list):
        min_val = min(C)
        C.remove(min_val)
        node = HuffmanTreeNode(None, None, min_val)
        return node

    def build_character(self, C : list, f : list, node : HuffmanTreeNode):
        try:
            index = f.index(int(node.f))
            f.pop(index)
            node.character = C[index]
            C.pop(index)
        except Exception as err:
            pass

    def huffman(self, C : list, f : list):
        '''
        赫夫曼编码

        算法自底向上的方式构造出最优编码所对应的树T

        Args
        ===
        `C` : 一个包含n个字符的集合，且每个字符都是一个出现频度为f[c]的对象

        '''
        n = len(f)
        Q = _deepcopy(f)
        tree = HuffmanTree()
        tree.characters = _deepcopy(self.C)
        tree.fs = _deepcopy(self.f)
        index = 0
        for i in range(n - 1):
            x = self.extract_min(Q)
            self.build_character(C, f, x)
            x.index = index
            index += 1
            y = self.extract_min(Q)
            self.build_character(C, f, y)
            y.index = index
            index += 1
            z = HuffmanTreeNode(x, y, x.f + y.f)
            z.index = index
            index += 1
            x.p = z
            y.p = z
            tree.addnode(z)
            Q.append(z.f)
        tree.renewall()
        tree.root = z
        tree.renewallcoding(tree.characters)
        
        return tree

    def build(self):
        '''
        构造一个HuffmanTree
        '''
        return self.huffman(self.C, self.f)
    
```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
'''

Class Chapter17_1

Class Chapter17_2

Class Chapter17_3

Class Chpater17_4

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

class Chapter17_1:
    '''
    chpater17.1 note and function
    '''
    def multipop(self, S : list, k):
        '''
        栈的弹出多个数据的操作
        '''
        while len(S) > 0 and k > 0:
            S.pop()
            k = k - 1

    def increment(self, A : list):
        i = 0
        while i < len(A) and A[i] == 1:
            A[i] = 0
            i += 1
        if i < len(A):
            A[i] = 1

    def note(self):
        '''
        Summary
        ====
        Print chapter17.1 note

        Example
        ====
        ```python
        Chapter17_1().note()
        ```
        '''
        print('chapter17.1 note as follow')  
        # !在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的
        print('在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的')
        # !平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能
        print('平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能')
        # !平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法
        print('平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法')
        print('17.1 聚集分析')
        print('在聚集分析中，要证明对所有的n,由n个操作所构成的序列的总时间在最坏的情况下为T(n).')
        print('因此，在最坏情况下，每个操作的平均代价(或称平摊代价)为T(n)/n')
        print('这个平摊代价对每个操作都是成立的，即使当序列中存在几种类型的操作也是一样的')
        print('例1.栈操作')
        print(' 10.1介绍了两种基本的栈操作，每种操作的时间代价都是O(1)：PUSH和POP操作')
        print(' 因此，含n个PUSH和POP操作的序列的总代价为n,而这n个操作的实际运行时间就是Θ(n)')
        print(' 现在增加一个栈操作MULTIPOP(S,k),它的作用使弹出栈S的k个栈顶对象，或者当栈包含少于k个对象时，弹出整个栈中的数据对象')
        print(' 分析一个由n个PUSH，POP和MULTIPOP操作构成的序列，其作用于一个初始为空的栈。')
        print(' 序列中一次MULTIPOP操作的最坏情况代价为O(n),因为栈的大小至多为n')
        print(' 因此，任意栈操作的最坏情况就是O(n),因此n个操作的序列的代价是O(n^2),因为可能会有O(n)个MULTIPOP操作，每个的代价都是O(n)')
        print(' 利用聚集分析，可以获得一个考虑到n个操作的整个序列的更好的上界')
        print(' 对任意的n值，包含n个PUSH,POP和MULTIPOP操作的序列的总时间为O(n).每个操作的平均代价为O(n)/n=O(1)')
        print(' 把每个操作的平摊代价指派为平均代价。在这个例子中，三个栈操作的平摊代价都是O(1)')
        print('例2.二进制计数器递增1')
        print(' 作为聚集分析的另一个例子，考虑实现一个由0开始向上计数的k位二进制计数器的问题')
        print(' 使用一个位数组A[0..k-1]作为计数器。存储在计数器中的一个二进制数x的最低位在A[0]中，最高位在A[k-1]')
        print(' 每次INCREMENT操作的代价都与被改变值的位数成线性关系')
        print(' 如同栈的例子，大致的分析只能得到正确但不紧确的界')
        print(' 在最坏的情况下，INCREMENT的每次执行要花Θ(nk)')
        print(' 注意到在每次调用INCREMENT时，并不是所有的位都翻转，可以分析得更紧确一些')
        print(' 来得到n次INCREMENT操作的序列的最坏情况代价为O(n)')
        print(' 在每次调用INCREMENT时，A[0]确实都要发生翻转，下一个高位A[1]每隔一次翻转',
            '当作用于初始为零的计数器上时，n次INCREMENT操作会导致A[1]翻转[n/2]次')
        print(' 对于i>[lgn],位A[i]始终保持不变。在序列中发生的位翻转的总次数为2n')
        print(' 所以在最坏的情况下，作用于一个初始为零的计数器上的n次INCREMENTC操作的时间为O(n)。')
        print(' 每次操作的平均代价(即每次操作的平摊代价)是O(n)/n=O(1)')
        print(' 练习17.1-1 如果一组栈操作中包括了一次MULTIPUSH操作，它一次把k个元素压入栈内，',
            '那么栈操作的平摊代价的界O(1)还能够保持')
        print(' 练习17.1-2 证明：在k位计数器的例子中，如果包含一个DECREMENT操作，n个操作可能花费Θ(nk)的时间')
        print(' 练习17.1-3 最坏情况是这个数据结构的每个数据都是2的整数幂，运行时间为O(n),聚集分析的每次操作的平摊代价为O(n)/n=1')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_2:
    '''
    chpater17.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.2 note

        Example
        ====
        ```python
        Chapter17_2().note()
        ```
        '''
        print('chapter17.2 note as follow')
        print('17.2 记账方法')
        print('在平摊分析的记账方法中。对不同的操作赋予不同的费用，某些操作的费用比它们的实际代价或多或少')
        print('对一个操作的收费的数量称为平摊代价')
        print('当一个操作的平摊代价超过了它的实际代价时，两者的差值就被当做存款(credit),并赋予数据结构中的一些特定对象')
        print('存款可以在以后用于补偿那些平摊代价低于其实际代价的操作')
        # !就可以将一个操作的平摊代价看做两部分：其实际代价与存款
        print('就可以将一个操作的平摊代价看做两部分：其实际代价与存款')
        print('记账方法与聚集方法有着很大的不同，对后者而言，所有操作都具有相同的平摊代价')
        print('选择操作的平摊代价必须很小心。如果希望通过对平摊代价的分析来说明每次操作的最坏情况平均代价较小,',
            '则操作序列的总的平摊代价就必须是该序列的总的实际代价的一个上界')
        print('记账方法和聚集方法一样，这种关系必须对所有的操作序列都成立')
        print('例1.栈操作')
        print(' 对于栈操作例子的平摊分析的记账方法，各个栈操作的实际代价为')
        print('  PUSH 1;')
        print('  POP 1;')
        print('  MULTIPOP min(k, s)')
        print(' 其中k为MULTIPOP的一个参数,s为调用该操作时栈的大小。现在对它们赋值以下的平摊代价')
        print('  PUSH 2;')
        print('  POP 0;')
        print('  MULTIPOP 0;')
        print(' 注意MULTIPOP的平摊代价是常数(0),而它的实际代价却是个变量,此处所有三个平摊代价都是O(1)')
        print(' 但一般来说，从渐进的意义上看，所考虑的各种操作的平摊代价是会发生变化的')
        print('结论：对任意的包含n次PUSH，POP和MULTIPOP操作的序列，总的平摊代价就是其总的实际代价的一个上界')
        print(' 又因为总的平摊代价为O(n),故总的实际代价为O(n)')
        print('例2.二进制计数器递增1')
        # !二进制计数器递增1这个操作的运行时间与发生翻转的位数是成正比的
        print(' 位数在本例中将被用作代价。用1元钱表示单位代价(即某一位的翻转)')
        print(' 因为计数器中为1的位数始终是非负的，故存款的总额总是非负的')
        print('因此对n次INCREMENT操作，总的平摊代价为O(n),这就给出了总的实际代价的一个界')
        print('练习17.2-1 对一个大小始终不超过k的栈上执行一系列栈操作。在每k个操作后，复制整个栈的内容以留作备份')
        print(' 证明：在对各种栈操作赋予合适的平摊代价后，n个栈操作(包括复制栈的操作)的代价为O(n)')
        print('练习17.2-2 略')
        print('练习17.2-3 假设希望不仅能使一个计数器增值，也能使之复位至零',
            '如何将一个计数器实现为一个位数组，使得对一个初始为零的计数器，',
            '任一个包含n个INCREMENT和RESET操作的序列的时间为O(n)')
        print(' 可以保持一个指针指向高位1')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_3:
    '''
    chpater17.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.3 note

        Example
        ====
        ```python
        Chapter17_3().note()
        ```
        '''
        print('chapter17.3 note as follow')
        print('17.3 势能方法')
        # !在平摊分析中，势能方法不是将已预付的工作作为存储在数据结构特定对象中的存款来表示，
        # !而是表示成一种“势能”或“势”，在需要时可以释放出来，以支付后面的操作。势是与整个数据结构而不是其中的个别对象发生联系的
        print('在平摊分析中，势能方法不是将已预付的工作作为存储在数据结构特定对象中的存款来表示，')
        print('而是表示成一种“势能”或“势”，在需要时可以释放出来，以支付后面的操作。势是与整个数据结构而不是其中的个别对象发生联系的')
        print('不同的势函数可能会产生不同的平摊代价，但它们欧式实际代价的上界')
        print('在选择一个势函数时常要做一些权衡；可选用的最佳势函数的选择要取决于所需的时间界')
        print('势能方法的工作过程：开始时，先对一个初始数据结构D0执行n个操作')
        print('对每个i=1,2,...,n，设ci为第i个操作的实际代价，Di为对数据结构Di-1作用第i个操作的结果')
        print('每个操作的平摊代价为其实际代价加上由于该操作所增加的势')
        print('例1.栈操作')
        print('定义栈上的势函数Φ 为栈中对象的个数')
        print(' 三种栈操作中每一种的平摊代价都是O(1),这样包含n个操作的序列的总平摊代价就是O(n)')
        print(' 故n个操作的总平摊代价即为总的实际代价的一个上界。所以n个操作的最坏情况代价为O(n)')
        print('例2.二进制计数器递增1')
        print('定义第i次INCREMENT操作后计数器的势为bi，即第i次操作后计数器中1的个数')
        print('因为b0<=k,只要k=O(n),总的实际代价就是O(n)。如果执行了至少n=Ω(k)次INCREMENT操作')
        print('无论计数器中包含什么样的初始值，总的实际代价都是O(n)')
        print('练习17.3-1 假设有势函数Φ，使得对所有的i都有Φ(Di)>=Φ(D0),但是Φ(D0) ≠ 0.')
        print(' 证明：存在一个势函数Φ`，使得Φ`(D0) = 0, Φ`(D1) >= 0, 对所有i >= 1, 且用Φ`表示的平摊代价与用ΦB表示的平摊代价相同')
        print('练习17.3-2 用势能方法的分析重做练习17.1-3')
        print('练习17.3-3 考虑一个包含n个元素的普通儿茶最小堆数据结构')
        print('它支持最坏情况时间代价为O(lgn)的操作INSERT和EXTRACT-MIN.请给出一个势函数Φ，使得INSERT的平摊代价为O(lgn)')
        print('EXTRACT-MIN的平摊代价为O(1),并证明函数确实是有用的')
        print('练习17.3-4 假设某个栈在执行n个操作PUSH、POP和MULTIPOP之前包含s0个对象，结束后包含sn个对象')
        print('练习17.3-5 假设一个计数器的二进制表示中在开始时有b个1，而不是0。证明：如果n=Ω(b)')
        print(' 则执行n次INCREMENT操作的代价为O(n) (不能假设b是常数)')
        print('练习17.3-6 说明如何用两个普通的栈来实现一个队列，使得每个ENQUEUE和DEQUEUE操作的平摊代价都为O(1)')
        print('练习17.3-7 设计一个数据结构来支持整数集合S上的下列两个操作')
        print(' INSERT(S, x) 将x插入S中')
        print(' DELETE_LARGER_HALF(S) 删除S中最大的[S/2]个元素')
        print('解释如何实现这个数据结构，使得任意m个操作的序列在O(m)时间内运行')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_4:
    '''
    chpater17.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.4 note

        Example
        ====
        ```python
        Chapter17_4().note()
        ```
        '''
        print('chapter17.4 note as follow')
        print('17.4 动态表')
        print('在某些应用中，无法预知要在表中存储多少个对象。为表分配了一定的空间，但后来发现空间并不够用')
        print('表的动态扩张和收缩问题。利用平摊分析，要证明插入和删除操作的平摊代价仅为O(1)')
        print('即使当它们引起了表的扩张和收缩时具有较大的实际代价也如此。')
        print('还可以看到如何来保证动态表中未用的空间始终不能超过整个空间的一个常量部分')
        print('使用在分析散列技术时的概念。定义一个非空表T的装载因子a(T)为表中存储的元素个数除以表的大小(槽的个数)后的结果')
        print('对一个空表(其中没有元素)定义其大小为0，其装载因子为1.')
        print('如果某一个动态表的装载银子以一个常数为上界，则表中未使用的空间就始终不会超过整个空间的一个常数部分')
        print('17.4.1 表扩张')
        print('假设一个表的存储空间分配为一个槽的数组。当所有的槽都被占用时，或者等价地，当其装载因子为1时，一个表就被填满了')
        print('当向一个满的表中插入一个项时，就能对原表进行扩张，即分配一个包含比原表更多槽的新表')
        print('因为总是需要这个表驻留在一段连续的内存中，必须为更大的表分配一个新的数组，然后把旧表中的项复制到新表中')
        print('一个常用的启发式方法是分配一个是原表两倍槽数的新表，如果只执行插入操作，则表的装载银子始终至少为1/2,浪费的空间就始终不会超过表空间的一半')
        print('这种启发式方法的特点是每当插入元素表被填满时，就重新分配两倍原来表空间的大小')
        print('如开始时表空间为1，当表满时，分配空间2，再满时分配空间4，以此类推，分配空间8，16，32')
        print('所以表扩张的程序的运行时间还由表中原来的元素个数决定')
        print('如果执行了n次操作，则依次操作的最坏情况代价为O(n),由此可得n次操作的总的运行时间的上界为O(n^2)')
        print('但是这个界不紧确，因为在执行n次插入操作的过程中，并不经常包括扩张表的代价')
        print('特别地，仅当i-1为2的整数幂时，第i次操作才会引起一次表的扩张')
        print('一次操作的平摊代价为O(1)')
        print('n次插入操作的总代价为3n,故每一次操作的平摊代价为3')
        print('用聚集方法，记账方法，势能方法分析')
        print('17.4.2 表扩张和收缩')
        print('为了实现TABLE_DELETE操作，只要将指定的项从表中去掉即可')
        print('但是当表的装载因子过小时，希望对表进行收缩，使得浪费的空间不致太大')
        print('表收缩与表扩张是类似的')
        print('当表中的项数降的过低时，就要分配一个新的、更小的表，而后将旧表的各项复制到新表中。')
        print('理想情况下，希望下面两个性质成立')
        print(' 动态表的装载因子由一个常数从下方限界')
        print(' 表操作的平摊代价由一个常数从上方限界')
        print('用基本插入和删除操作来度量代价')
        print('关于表收缩和扩张的一个自然的策略是当向满表中插入一个项时，将表的规模扩大一倍，',
            '而当从表中删除一个项就导致表不足半满')
        print('总之，因为每个操作的平摊代价都有一个常数上界，所以作用于一动态表上的n个操作的实际时间为O(n)')
        print('练习17.4-1 希望实现一个动态开放地址散列表。当表的装载因子达到某个严格小于1的数a时，就可以认为表满了')
        print(' 因为表的空间大小是一个有限的实数，并不是无限的(比如受内存的限制),所以当表还差一个元素未满时，装载因子此时严格小于1')
        print(' 说明如何对一个动态开放地址散列表进行插入，使得每个插入操作的平摊代价的期望值为O(1),',
            '为什么每个插入操作的实际代价的期望值不必是O(1)')
        print('练习17.4-2 证明：如果作用于一动态上的第i个操作是TABLE-DELETE,且装载因子大于0.5时')
        print(' 则以势函数表示的每个操作的平摊代价由一个常数从上方限界')
        print('练习17.4-3 假设当某个表的装载因子下降至1/4一下时，不是通过将其大小缩小一半来收缩')
        print(' 而是在表的装载因子低于1/3时，通过将其大小乘以2/3来进行收缩')
        print(' 利用势函数证明采用这种策略的TABLE-DELETE操作的平摊代价由一个常数从上方限界')
        print('思考题17-1 位反向的二进制计数器')
        # !FFT算法的第一步在输入数组A[0..n-1]上执行一次位反向置换
        print('FFT算法的第一步在输入数组A[0..n-1]上执行一次位反向置换')
        print('其中数组长度n=2^k(k为非负整数)。这个置换将某些元素进行交换，这些元素的下标的二进制表示彼此相反')
        print('revk(3)=12,revk(1)=8,revk(2)=4')
        print('比如n=16,k=4,写出一个能在O(nk)时间内完成对一个长度为n=2^k(k为非负整数)的数组的位反向置换算法')
        print('可以用一个基于平摊分析的算法来改善位反向置换的运行时间，方法是采用一个\"位反向计数器\"和一个程序',
            'BIT-REVERSED-INCREMENT该程序在给定一个位反向计数器和一个程序')
        print('如果k=4，且计数器的初始值位0，则连续调用BIT-REVERSED-INCREMENT就产生序列')
        print('0000,1000,0100,1100,0010,1010,...=0,8,4,12,2,10')
        print('假设可以在单位时间内对一个字左移或右移一位，可以实现一个O(n)时间的位反向置换')
        print('思考题17-2 使二叉查找动态化')
        # !对一个已经排序好的数组的二叉查找要花对数的时间
        print('对一个已经排序好的数组的二叉查找要花对数的时间,而插入一个新元素的时间则与数组的大小成线性关系')
        print('通过使用若干排好序的数组，就可以改善插入的时间')
        print('思考题17-3 平摊加权平衡树')
        print('假设在一棵普通的二叉查找树中，对每个结点x增加一个域size[x],',
            '表示在以x为根的子树中关键字的个数')
        print('设a是在范围1/2<=a<1之间的一个常数。')
        print('如果a * x.size >= x.left.size and a * x.size >= x.right.size',
            '就说一给定的结点x是a平衡的')
        print('如果树中每个结点都是a平衡的，则整棵树就是a平衡的')
        print('在最坏的情况下，对一棵有n个结点，a平衡的二叉查找树做一次查找要花O(lgn)时间')
        print('任何二叉查找树都具有非负的势；一棵1/2平衡树的势为0')
        print('对一棵包含n个结点的a平衡树，插入一个结点或删除一个结点需要O(lgn)平摊时间')
        print('思考题17-4 重构红黑树的代价')
        # !在红黑树上，有4种基本操作会做结构性的修改，它们是结点插入，结点删除、旋转以及颜色修改操作
        print('在红黑树上，有4种基本操作会做结构性的修改，它们是结点插入，结点删除、旋转以及颜色修改操作')
        print('描述一棵有n个结点的合法红黑树,使得调用RB-INSERT来插入第(n+1)个结点会引起Ω(lgn)次颜色修改')
        print('描述一棵有n个结点的合法红黑树，使得在一个特殊结点上调用RB-DELETE会引起Ω(lgn)次颜色修改')
        print('在最坏情况下，任何有m个RB-INSERT和RB-DELETE操作的序列执行O(m)次结构修改')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

chapter17_1 = Chapter17_1()
chapter17_2 = Chapter17_2()
chapter17_3 = Chapter17_3()
chapter17_4 = Chapter17_4()

def printchapter17note():
    '''
    print chapter17 note.
    '''
    print('Run main : single chapter seventeen!')  
    chapter17_1.note()
    chapter17_2.note()
    chapter17_3.note()
    chapter17_4.note()

# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
if __name__ == '__main__':  
    printchapter17note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter18/chapter18note.py
# python3 src/chapter18/chapter18note.py
'''

Class Chapter18_1

Class Chapter18_2

Class Chapter18_3

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

if __name__ == '__main__':
    import btree as bt
else:
    from . import btree as bt

class Chapter18_1:
    '''
    chpater18.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.1 note

        Example
        ====
        ```python
        Chapter18_1().note()
        ```
        '''
        print('chapter18.1 note as follow')  
        print('第五部分 高级数据结构')
        # !B树是一种被设计成专门存储在磁盘上的平衡查找树
        print('第18章 B树是一种被设计成专门存储在磁盘上的平衡查找树')
        print(' 因为磁盘的操作速度要大大慢于随机存取存储器，所以在分析B树的性能时')
        print(' 不仅要看动态集合操作花了多少计算时间，还要看执行了多少次磁盘存取操作')
        print(' 对每一种B树操作，磁盘存取的次数随B树高度的增加而增加，而各种B树操作又能使B树保持较低的高度')
        print('第19章,第20章 给出可合并堆的几种实现。')
        print(' 这种堆支持操作INSERT,MINIMUM,EXTRACT-MIN和UNION.')
        print(' UNION操作用于合并两个堆。这两章中出现的数据结构还支持DELETE和DECREASE-KEY操作')
        print('第19章中出现的二项堆结构能在O(lgn)最坏情况时间内支持以上各种操作，此处n位输入堆中的总元素数')
        print('第20章 斐波那契堆对二项堆进行了改进 操作INSERT,MINIMUM和UNION仅花O(1)的实际和平摊时间')
        print(' 操作EXTRACT-MIN和DELETE要花O(lgn)的平摊时间')
        # !渐进最快的图问题算法中，斐波那契堆是其核心部分
        print(' 操作DECREASE-KEY仅花O(1)的平摊时间')
        print('第21章 用于不想交集合的一些数据结构，由n个元素构成的全域被划分成若干动态集合')
        print(' 一个由m个操作构成的序列的运行时间为O(ma(n)),其中a(n)是一个增长的极慢的函数')
        print(' 在任何可想象的应用中，a(n)至多为4.')
        print(' 这个问题的数据结构简单，但用来证明这个时间界的平摊分析却比较复杂')
        print('其他一些高级的数据结构：')
        print(' 动态树：维护一个不相交的有根树的森林')
        print('  在动态树的一种实现中，每个操作具有O(lgn)的平摊时间界；',
            '在另一种更复杂的实现中，最坏情况时间界O(lgn).动态树常用在一些渐进最快的网络流算法中')
        print(' 伸展树：是一种二叉查找树，标准的查找树操作在其上以O(lgn)的平摊时间运行,',
            '伸展树的一个应用是简化动态树')
        print(' 持久的数据结构允许在过去版本的数据结构上做查询，甚至有时候做更新,',
            '只需很小的时空代价，就可以使链式数据结构持久化的技术')
        print('第18章 B 树')
        # !B树是为磁盘或其他直接存取辅助设备而设计的一种平衡查找树
        print('B树是为磁盘或其他直接存取辅助设备而设计的一种平衡查找树。与红黑树类似，',
            '但是在降低磁盘I/O操作次数方面更好一些。许多数据库系统使用B树或者B树的变形来存储信息')
        # !B树与红黑树的主要不同在于，B树的结点可以有许多子女，从几个到几千个，就是说B树的分支因子可能很大
        print('B树与红黑树的主要不同在于，B树的结点可以有许多子女，从几个到几千个，就是说B树的分支因子可能很大')
        print('这一因子常常是由所使用的磁盘特性所决定的。')
        print('B树与红黑树的相似之处在于，每棵含有n个结点的B树高度为O(lgn),',
            '但可能要比一棵红黑树的高度小许多，因为分支因子较大')
        print('所以B树也可以被用来在O(lgn)时间内，实现许多动态集合操作')
        print('B树以自然的方式推广二叉查找树。如果B树的内结点x包含x.n个关键字，则x就有x.n+1个子女')
        print('结点x中的关键字是用来将x所处理的关键字域划分成x.n+1个子域的分隔点，每个子域都由x中的一个子女来处理')
        print('铺存上的数据结构')
        print(' 有许多不同的技术可用来在计算机中提供存储能力')
        print(' 典型的磁盘驱动器，这个驱动器包含若干盘片，它们以固定速度绕共用的主轴旋转')
        print('虽然磁盘比主存便宜而且有搞笑的容量，但是它们速度很慢')
        print('有两种机械移动的成分：盘旋转和磁臂移动')
        print('在一个典型的B树应用中，要处理的数据量很大，因此无法一次都装入主存')
        print('B树算法将所需的页选择出来复制到主存中去，而后将修改过的页再写回到磁盘上去')
        print('18.1 B树的定义')
        print('一棵B树T是具有如下性质的有根树(根为root[T]):')
        print('1) 每个结点x有以下域')
        print(' a) n[x],当前存储在结点x中的关键字数')
        print(' b) n[x]个关键字本身，以非降序存放，因此key1[x]<=key2[x]<=...<=keyn[x]')
        print(' c) leaf[x],是一个布尔值，如果x是叶子的话，则它为TRUE,如果x为一个内结点，则为FALSE')
        print('2) 每个内结点x还包含n[x]+1个指向其子女的指针c1[x],c2[x],...,cn[x]+1[x].叶结点没有子女，故它们的ci域无定义')
        print('3) 各关键字keyi[x]对存储在各子树中的关键字范围加以分隔：如果ki为存储在以ci[x]为根的子树中的关键字')
        print('4) 每个叶结点具有相同的深度，即树的高度h')
        print('5) 每一个节点能包含的关键字数有一个上界和下界。这些界可用一个称作B树的最小度数的固定整数t>=2来表示')
        print(' a) 每个非根的结点必须至少有t-1个关键字。每个非根的内结点至少有t个子女。',
            '如果树是非空的，则根节点至少包含一个关键字')
        print(' b) 每个结点可包含至多2t-1个关键字。所以一个内结点至多可有2t个子女。')
        print('   如果某结点恰好2t-1个关键字，则根节点至少包含一个关键字')
        print('t=2时的B树(二叉树)是最简单的。这时每个内结点有2个、3个或者4个子女，亦即一棵2-3-4树。',
            '然而在实际中，通常是采用大得多的t值')
        print('B树的高度')
        print(' B树上大部分操作所需的磁盘存取次数与B树的高度成正比')
        print('定理18.1 如果n>=1,则对任意一棵包含n个关键字、高度为h、最小度数t>=2的B树T，有：h<=logt((n + 1) / 2)')
        print('证明：如果一棵B树的高度为h,其根结点包含至少一个关键字而其他结点包含至少t-1个关键字。',
            '这样，在深度1至少有两个结点，在深度2至少有2t个结点，',
                '在深度3至少有2t^2个结点，直到深度h至少有2t^(h-1)个结点。')
        print('与红黑树和B树相比，虽然两者的高度都以O(lgn)的速度增长，对B树来说对数的底要大很多倍')
        print('对大多数的树操作来说，要查找的结点数在B树中要比在红黑树中少大约lgt个因子')
        print('因为在树中查找任意一个结点通常都需要一次磁盘存取，所以磁盘存取的次数大大地减少了')
        print('练习18.1-1 为什么不允许B树的最小度数t为1，t最小只能取2(二叉树)，再小就构不成树了')
        print('练习18.1-2 t的取值为t>=2,才能使图中的树是棵合法的B树')
        print('练习18.1-3 二叉树，二叉查找树，2-3-4树都可以成为最小度数为2的所有合法B树')
        print('练习18.1-4 由定理18.1中的公式h<=logt((n + 1) / 2)得，',
            '一棵高度为h的B树中，可以最多存储[2t^h-1]个结点')
        print('练习18.1-5 如果红黑树中的每个黑结点吸收它的红子女，并把它们的子女并入自身，描述这个结果的数据结构')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_2:
    '''
    chpater18.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.2 note

        Example
        ====
        ```python
        Chapter18_2().note()
        ```
        '''
        print('chapter18.2 note as follow')
        print('18.2 对B树的基本操作')
        print('这一节给出操作B-TREE-SEARCH, B-TREE-CREATE和B-TREE-INSERT的细节，两个约定:')
        print(' (1) B树的根结点始终在主存中，因而无需对根做DISK-DEAD；但是，',
            '每当根结点被改变后，都需要对根结点做一次DISK-WRITE')
        print(' (2) 任何被当做参数的结点被传递之前，要先对它们做一次DISK-READ')
        print('给出的过程都是\"单向\"算法，即它们沿树的根下降，没有任何回溯')
        print('搜索B树')
        print(' 搜索B树有搜索二叉查找树很相似，只是不同于二叉查找树的两路分支，而是多路分支')
        print(' 即在每个内结点x处，要做x.n+1路的分支决定')
        print(' B-TREE-SEARCH是定义在二叉查找树上的TREE-SEARCH过程的一个直接推广。',
            '它的输入是一个指向某子树的根结点x的指针，以及要在该子树中搜索的一个关键字k',
            '顶层调用的形式为B-TREE-SEARCH(root, key).如果k在B树中，',
            'B-TREE-SEARCH就返回一个由结点y和使keyi[y]==k成立的下标i组成的有序对(y, i)',
            '否则返回NONE')
        print(' 像在二叉查找树的TREE-SEARCH过程中那样，在递归过程中所遇到的结点构成以一条从树根下降的路径')
        print('创建一棵空的B树')
        print(' 为构造一棵B树T，先用B-TREE-CREATE来创建一个空的根结点，再调用B-TREE-INSERT来加入新的关键字')
        print('向B树插入关键字')
        print(' 与向二叉查找树中插入一个关键字相比向B树中插入一个关键字复杂得多。')
        print(' 像在二叉查找树中一样，要查找插入新关键字的叶子位置。',
            '但是在B树中，不能简单地创建一个新的叶结点，然后将其插入,因为这样得到的树不再是一颗有效的B树')
        print('B树中结点的分裂')
        print(' 过程B-TREE-SPLIT-CHILD的输入一个非满的内结点x(假定在主存当中)',
            '下标i以及一个结点y(同样假定在其主存当中)，y=x.c[i]是x的一个满子结点')
        print(' 该过程把这个孩子分裂两个，并调整x使之有一个新增的孩子')
        print(' 要分裂一个满的根，首先让根成为一个新的空跟结点的孩子，',
            '这样才能够使用B-TREE-SPLIT-CHILD，树的高度因此增加1，分裂是树长高的唯一途径')
        print('对B树用单程下行便利树方式插入关键字')
        print('对一棵高度为h的B树，B-TREE-INSERT要做的磁盘存取次数为O(h),因为在调用B-TREE-INSERT-NONFULL之间',
            '只做了O(1)次DISK-READ和DISK-WRITE操作，所占用的总的CPU时间为O(th)=O(tlogt(n))')
        print('因为B-TREE-INSERT-NONFULL是尾递归的，故也可以用一个while循环来实现')
        print('说明了在任何时刻，需要留在主存中的页面数为O(1)')
        keys = ['F', 'S', 'Q', 'K', 'C', 'L', 'H', 'T', 'V', \
            'W', 'M', 'R', 'N', 'P', 'A', 'B', 'X', 'Y', 'D', 'Z', 'E']
        btree = bt.BTree(2)
        for key in keys:
            btree.insert(key)
        print('练习18.2-1 请给出关键字', keys, '依照顺序插入一棵最小度数为2的空的B树的结果')
        print(btree.root)
        print('练习18.2-2 在child_split的情况下，在调用B-TREE-INSERT的过程中',
            '会执行冗余的DISK-READ或DISK-WRITE')
        print(' 所谓冗余的DISK-READ是指')
        print('练习18.2-3 前驱从最大子结点的最大关键字寻找，后继从最小自结点的最小关键字寻找')
        print('练习18.2-4 假设关键字{1,2,...,n}被插入一个最小度数为2的空B树中。最终的B树有多少结点')
        btree = bt.BTree(2)
        for i in range(1, 12):
            btree.insert(i)
        print(btree.root)
        btree.display()
        print(' 由于每次插入的树的关键字都比前面的字大，因此新关键字永远是放到了最右边的结点中')
        print(' 除了最右边一直往下的路径上的结点(记为R)中的关键字数有可能大于1外')
        print(' 其他所有节点的关键字数量都是1，当所有的R结点都有三个关键字时，有最少的结点数')
        print(' 此时n=2^(h+1)-1+2(h+1),其中h是B树的高度，结点数2^(h+1)-1.')
        print(' 而2^(h+1)-1=n-2(h+1),其中h=Θ(lgn),因此结点数为Θ(n)')
        print('练习18.2-5 因为叶子结点无需指向子女的指针，对同样大小的磁盘页')
        print(' 可选用一个与内结点不同的(更大的)t值')
        print('练习18.2-6 假设B-TREE-SEARCH的实现是在每个结点处采用二叉查找，而不是线性查找。')
        print(' 证明无论怎样选择t(t为n的函数)。这种实现所需的CPU时间都为O(lgn)')
        print('练习18.2-7 假设磁盘硬件允许任意选择磁盘页的大小，但读取磁盘页的时间为a+bt')
        print(' 其中a和b为规定的常数，t为确定磁盘页的大小后，B树的最小度数')
        print(' 请描述如何选择t以最小化B树的查找时间。对a=5毫秒和b=10微秒，请给出t的一个最优值')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

class Chapter18_3:
    '''
    chpater18.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter18.3 note

        Example
        ====
        ```python
        Chapter18_3().note()
        ```
        '''
        print('chapter18.3 note as follow')
        print('从B树中删除关键字')
        print(' 删除操作与插入操作类似，只是略微复杂，因为一个关键字可以从任意一个结点删除，而不只是从叶子中删除')
        print(' 从一个内部结点删除子女时，需要重新安排这个结点的子女')
        print(' 防止因删除操作影响B树的性质')
        print('练习18.3-1 略')
        print('练习18.3-2 略')
        print('思考题18-1 辅存上的栈')
        print('思考题18-2 连接与分裂2-3-4树')
        # python src/chapter18/chapter18note.py
        # python3 src/chapter18/chapter18note.py

chapter18_1 = Chapter18_1()
chapter18_2 = Chapter18_2()
chapter18_3 = Chapter18_3()

def printchapter18note():
    '''
    print chapter18 note.
    '''
    print('Run main : single chapter eighteen!')  
    chapter18_1.note()
    chapter18_2.note()
    chapter18_3.note()

# python src/chapter18/chapter18note.py
# python3 src/chapter18/chapter18note.py
if __name__ == '__main__':  
    printchapter18note()
else:
    pass

```

```py

class BTreeNode:
    '''
    B树结点
    '''
    def __init__(self, n = 0, isleaf = True):
        '''
        B树结点

        Args
        ===
        `n` : 结点包含关键字的数量

        `isleaf` : 是否是叶子节点

        '''
        # 结点包含关键字的数量
        self.n = n
        # 关键字的值数组
        self.keys = []
        # 子结点数组
        self.children = []
        # 是否是叶子节点
        self.isleaf = isleaf

    def __str__(self):

        returnStr = 'keys:['
        for i in range(self.n):
            returnStr += str(self.keys[i]) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']\r\n'
        return returnStr

    def diskread(self):
        '''
        磁盘读
        '''
        pass

    def diskwrite(self):
        '''
        磁盘写
        '''
        pass

    @classmethod
    def allocate_node(self, key_max):
        '''
        在O(1)时间内为一个新结点分配一个磁盘页

        假定由ALLOCATE-NODE所创建的结点无需做DISK-READ，因为磁盘上还没有关于该结点的有用信息

        Return
        ===
        `btreenode` : 分配的B树结点

        Example
        ===
        ```python
        btreenode = BTreeNode.allocate_node()
        ```
        '''
        node = BTreeNode()
        child_max = key_max + 1
        for i in range(key_max):
            node.keys.append(None)
        for i in range(child_max):
            node.children.append(None)
        return node

class BTree:
    '''
    B树
    '''
    def __init__(self, m = 3):
        '''
        B树的定义
        '''
        # B树的最小度数
        self.M = m
        # 节点包含关键字的最大个数
        self.KEY_MAX = 2 * self.M - 1
        # 非根结点包含关键字的最小个数
        self.KEY_MIN = self.M - 1
        # 子结点的最大个数
        self.CHILD_MAX = self.KEY_MAX + 1
        # 子结点的最小个数
        self.CHILD_MIN = self.KEY_MIN + 1
        # 根结点
        self.root: BTreeNode = None

    def __new_node(self):
        '''
        创建新的B树结点
        '''
        return BTreeNode.allocate_node(self.KEY_MAX)

    def insert(self, key):
        '''
        向B树中插入新结点`key`  
        '''
        # 检查关键字是否存在
        if self.contain(key) == True:
            return False
        else:
            # 检查是否为空树
            if self.root is None:
                node = self.__new_node()
                node.diskwrite()
                self.root = node    
            # 检查根结点是否已满      
            if self.root.n == self.KEY_MAX:
                # 创建新的根结点
                pNode = self.__new_node()
                pNode.isleaf = False
                pNode.children[0] = self.root
                self.__split_child(pNode, 0, self.root)
                # 更新结点指针
                self.root = pNode
            self.__insert_non_full(self.root, key)
            return True

    def remove(self, key): 
        '''
        从B中删除结点`key`
        '''      
        # 如果关键字不存在
        if not self.__search(self.root, key):
            return False
        # 特殊情况处理
        if self.root.n == 1:
            if self.root.isleaf == True:
                self.clear()
            else:
                pChild1 = self.root.children[0]
                pChild2 = self.root.children[1]
                if pChild1.n == self.KEY_MIN and pChild2.n == self.KEY_MIN:
                    self.__merge_child(self.root, 0)
                    self.__delete_node(self.root)
                    self.root = pChild1
        self.__recursive_remove(self.root, key)
        return True
    
    def display(self):
        '''
        打印树的关键字  
        '''
        self.__display_in_concavo(self.root, self.KEY_MAX * 10)

    def contain(self, key):
        '''
        检查该`key`是否存在于B树中  
        '''
        self.__search(self.root, key)

    def clear(self):
        '''
        清空B树  
        '''
        self.__recursive_clear(self.root)
        self.root = None

    def __recursive_clear(self, pNode : BTreeNode):
        '''
        删除树  
        '''
        if pNode is not None:
            if not pNode.isleaf:
                for i in range(pNode.n):
                    self.__recursive_clear(pNode.children[i])
            self.__delete_node(pNode)

    def __delete_node(self, pNode : BTreeNode):
        '''
        删除节点 
        '''
        if pNode is not None:
            pNode = None
    
    def __search(self, pNode : BTreeNode, key):
        '''
        查找关键字  
        '''
        # 检测结点是否为空，或者该结点是否为叶子节点
        if pNode is None:
            return False
        else:
            i = 0
            # 找到使key < pNode.keys[i]成立的最小下标
            while i < pNode.n and key > pNode.keys[i]:
                i += 1
            if i < pNode.n and key == pNode.keys[i]:
                return True
            else:
                # 检查该结点是否为叶子节点
                if pNode.isleaf == True:
                    return False
                else:
                    return self.__search(pNode.children[i], key)

    def __split_child(self, pParent : BTreeNode, nChildIndex, pChild : BTreeNode):
        '''
        分裂子节点
        '''
        # 将pChild分裂成pLeftChild和pChild两个结点
        pRightNode = self.__new_node()  # 分裂后的右结点
        pRightNode.isleaf = pChild.isleaf
        pRightNode.n = self.KEY_MIN
        # 拷贝关键字的值
        for i in range(self.KEY_MIN):
            pRightNode.keys[i] = pChild.keys[i + self.CHILD_MIN]
        # 如果不是叶子结点，就拷贝孩子结点指针
        if not pChild.isleaf:
            for i in range(self.CHILD_MIN):
                pRightNode.children[i] = pChild.children[i + self.CHILD_MIN]
        # 更新左子树的关键字个数
        pChild.n = self.KEY_MIN
        # 将父结点中的pChildIndex后的所有关键字的值和子树指针向后移动一位
        for i in range(nChildIndex, pParent.n):
            j = pParent.n + nChildIndex - i
            pParent.children[j + 1] = pParent.children[j]
            pParent.keys[j] = pParent.keys[j - 1]
        # 更新父结点的关键字个数
        pParent.n += 1
        # 存储右子树指针
        pParent.children[nChildIndex + 1] = pRightNode
        # 把结点的中间值提到父结点
        pParent.keys[nChildIndex] = pChild.keys[self.KEY_MIN]
        pChild.diskwrite()
        pRightNode.diskwrite()
        pParent.diskwrite()
    
    def __insert_non_full(self, pNode: BTreeNode, key):
        '''
        在非满节点中插入关键字
        '''
        # 获取结点内关键字个数
        i = pNode.n
        # 如果pNode是叶子结点
        if pNode.isleaf == True:
            # 从后往前 查找关键字的插入位置
            while i > 0 and key < pNode.keys[i - 1]:
                # 向后移位
                pNode.keys[i] = pNode.keys[i - 1]
                i -= 1
            # 插入关键字的值
            pNode.keys[i] = key
            # 更新结点关键字的个数
            pNode.n += 1
            pNode.diskwrite()
        # pnode是内结点
        else:
            # 从后往前 查找关键字的插入的子树
            while i > 0 and key < pNode.keys[i - 1]:
                i -= 1
            # 目标子树结点指针
            pChild = pNode.children[i]
            pNode.children[i].diskread()
            # 子树结点已经满了
            if pChild.n == self.KEY_MAX:
                # 分裂子树结点
                self.__split_child(pNode, i, pChild)
                # 确定目标子树
                if key > pNode.keys[i]:
                    pChild = pNode.children[i + 1]
            # 插入关键字到目标子树结点
            self.__insert_non_full(pChild, key)

    def __display_in_concavo(self, pNode: BTreeNode, count):
        '''
        用括号打印树 
        '''
        if pNode is not None:
            i = 0
            j = 0
            for i in range(pNode.n):
                if not pNode.isleaf:
                    self.__display_in_concavo(pNode.children[i], count - 2)
                for j in range(-1, count):
                    k = count - j - 1
                    print('-', end='')
                print(pNode.keys[i])
            if not pNode.isleaf:
                self.__display_in_concavo(pNode.children[i], count - 2)

    def __merge_child(self, pParent: BTreeNode, index):
        '''
        合并两个子结点
        '''
        pChild1 = pParent.children[index]
        pChild2 = pParent.children[index + 1]
        # 将pChild2数据合并到pChild1
        pChild1.n = self.KEY_MAX
        # 将父结点index的值下移
        pChild1.keys[self.KEY_MIN] = pParent.keys[index]
        for i in range(self.KEY_MIN):
            pChild1.keys[i + self.KEY_MIN + 1] = pChild2.keys[i]
        if not pChild1.isleaf:
            for i in range(self.CHILD_MIN):
                pChild1.children[i + self.CHILD_MIN] = pChild2.children[i]
        # 父结点删除index的key，index后的往前移一位
        pParent.n -= 1
        for i in range(index, pParent.n):
            pParent.keys[i] = pParent.keys[i + 1]
            pParent.children[i + 1] = pParent.children[i + 2]
        # 删除pChild2
        self.__delete_node(pChild2)

    def __recursive_remove(self, pNode: BTreeNode, key):
        '''
        递归的删除关键字`key`  
        '''
        i = 0
        while i < pNode.n and key > pNode.keys[i]:
            i += 1
        # 关键字key在结点pNode
        if i < pNode.n and key == pNode.keys[i]:
            # pNode是个叶结点
            if pNode.isleaf == True:
                # 从pNode中删除k
                for j in range(i, pNode.n):
                    pNode.keys[j] = pNode.keys[j + 1]
                return
            # pNode是个内结点
            else:
                # 结点pNode中前于key的子结点
                pChildPrev = pNode.children[i]
                # 结点pNode中后于key的子结点
                pChildNext = pNode.children[i + 1]
                if pChildPrev.n >= self.CHILD_MIN:
                    # 获取key的前驱关键字
                    prevKey = self.predecessor(pChildPrev)
                    self.__recursive_remove(pChildPrev, prevKey)
                    # 替换成key的前驱关键字
                    pNode.keys[i] = prevKey
                    return
                # 结点pChildNext中至少包含CHILD_MIN个关键字
                elif pChildNext.n >= self.CHILD_MIN:
                    # 获取key的后继关键字
                    nextKey = self.successor(pChildNext)
                    self.__recursive_remove(pChildNext, nextKey)
                    # 替换成key的后继关键字
                    pNode.keys[i] = nextKey
                    return
                # 结点pChildPrev和pChildNext中都只包含CHILD_MIN-1个关键字
                else:
                    self.__merge_child(pNode, i)
                    self.__recursive_remove(pChildPrev, key)
        # 关键字key不在结点pNode中
        else:
            # 包含key的子树根结点
            pChildNode = pNode.children[i]
            # 只有t-1个关键字
            if pChildNode.n == self.KEY_MAX:
                # 左兄弟结点
                pLeft = None
                # 右兄弟结点
                pRight = None
                # 左兄弟结点
                if i > 0:
                    pLeft = pNode.children[i - 1]
                # 右兄弟结点
                if i < pNode.n:
                    pRight = pNode.children[i + 1]
                j = 0
                if pLeft is not None and pLeft.n >= self.CHILD_MIN:
                    # 父结点中i-1的关键字下移至pChildNode中
                    for j in range(pChildNode.n):
                        k = pChildNode.n - j
                        pChildNode.keys[k] = pChildNode.keys[k - 1]
                    pChildNode.keys[0] = pNode.keys[i - 1]
                    if not pLeft.isleaf:
                        # pLeft结点中合适的子女指针移到pChildNode中
                        for j in range(pChildNode.n + 1):
                            k = pChildNode.n + 1 - j
                            pChildNode.children[k] = pChildNode.children[k - 1]
                        pChildNode.children[0] = pLeft.children[pLeft.n]
                    pChildNode.n += 1
                    pNode.keys[i] = pLeft.keys[pLeft.n - 1]
                    pLeft.n -= 1
                # 右兄弟结点至少有CHILD_MIN个关键字
                elif pRight is not None and pRight.n >= self.CHILD_MIN:
                    # 父结点中i的关键字下移至pChildNode中
                    pChildNode.keys[pChildNode.n] = pNode.keys[i]
                    pChildNode.n += 1
                    # pRight结点中的最小关键字上升到pNode中
                    pNode.keys[i] = pRight.keys[0]
                    pRight.n -= 1
                    for j in range(pRight.n):
                        pRight.keys[j] = pRight.keys[j + 1]
                    if not pRight.isleaf:
                        # pRight结点中合适的子女指针移动到pChildNode中
                        pChildNode.children[pChildNode.n] = pRight.children[0]
                        for j in range(pRight.n):
                            pRight.children[j] = pRight.children[j + 1]
                # 左右兄弟结点都只包含CHILD_MIN-1个结点
                elif pLeft is not None:
                    self.__merge_child(pNode, i - 1)
                    pChildNode = pLeft
                # 与右兄弟合并
                elif pRight is not None:
                    self.__merge_child(pNode, i)
            self.__recursive_remove(pChildNode, key)

    def predecessor(self, pNode: BTreeNode):
        '''
        前驱关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[pNode.n]
        return pNode.keys[pNode.n - 1]

    def successor(self, pNode: BTreeNode):
        '''
        后继关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[0]
        return pNode.keys[0]

def test():
    '''
    test class `BTree` and class `BTreeNode`
    '''
    tree = BTree(3)
    tree.insert(11)
    tree.insert(3)
    tree.insert(1)
    tree.insert(4)
    tree.insert(33)
    tree.insert(13)
    tree.insert(63)
    tree.insert(43)
    tree.insert(2)
    print(tree.root)
    tree.display()
    tree.clear()
    tree = BTree(2)
    tree.insert(11)
    tree.insert(3)
    tree.insert(1)
    tree.insert(4)
    tree.insert(33)
    tree.insert(13)
    tree.insert(63)
    tree.insert(43)
    tree.insert(2)
    print(tree.root)
    tree.remove(1)
    tree.remove(2)
    tree.remove(3)
    print(tree.root)
    tree.clear()
    print(tree.root)
    tree.display()

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
'''

Class Chapter19_1

Class Chapter19_2

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

if __name__ == '__main__':
    import binomialheap as bh
else:
    from . import binomialheap as bh

class Chapter19_1:
    '''
    chpater19.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter19.1 note

        Example
        ====
        ```python
        Chapter19_1().note()
        ```
        '''
        print('chapter19.1 note as follow')  
        print('第19章 二项堆')
        # !可合并堆(包括二叉堆、二项堆、斐波那契堆)的数据结构，这些数据结构支持下面五种操作
        print('可合并堆(包括二叉堆、二项堆、斐波那契堆)的数据结构，这些数据结构支持下面五种操作')
        print('MAKE-HEAP():创建并返回一个不包含任何元素的新堆')
        print('INSERT(H,x):将结点x(其关键字域中已填入了内容)插入堆H中')
        print('MINIMUM(H):返回一个指向堆H中包含最小关键字的结点的指针')
        print('EXTRACT-MIN(H):将堆H中包含的最小关键字删除，并返回一个指向该结点的指针')
        print('UNION(H1,H2):创建并返回一个包含堆H1和H2中所有结点的新堆。同时H1和H2被这个操作\"删除\"')
        print('DECREASE-KEY(H, x, k):将新关键字值k(假定它不大于当前的关键字值)赋给堆H中的结点x')
        print('DELETE(H, x):从堆H中删除结点x')
        print('    过程       |二叉堆(最坏情况)|二项堆(最坏情况)|斐波那契堆(平摊)|')
        print(' MAKE-HEAP()   |     Θ(1)      |      Θ(1)    |     Θ(1)      |')
        print(' INSERT(H,x)   |    Θ(lgn)     |     Ω(lgn)   |     Θ(1)      |')
        print(' MINIMUM(H)    |     Θ(1)      |     Ω(lgn)   |     Θ(1)      |')
        print(' EXTRACT-MIN(H)|    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print(' UNION(H1,H2)  |     Θ(n)      |     Ω(lgn)   |     Θ(1)      |')
        print(' DECREASE-KEY  |    Θ(lgn)     |     Θ(lgn)   |     Θ(1)      |')
        print(' DELETE(H, x)  |    Θ(lgn)     |     Θ(lgn)   |    O(lgn)     |')
        print('对操作SEARCH操作的支持方面看，二叉堆、二项堆、斐波那契堆都是低效的')
        print('19.1 二项树和二项堆')
        print('19.1.1 二项树')
        # !二项树Bk是一种递归定义的树。
        print('二项树Bk是一种递归定义的树。')
        # !二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子
        print('二项树B0只含包含一个结点。二项树Bk由两颗二项树Bk-1链接而成：其中一棵树的根的是另一棵树的根的最左孩子')
        print('引理19.1(二项树的性质) 二项树Bk具有以下的性质')
        print('1) 共有2^k个结点')
        print('2) 树的高度为k')
        print('3) 在深度i处恰有(k i)个结点，其中i=0,1,2,...,k')
        print('4) 根的度数为k，它大于任何其他结点的度数；',
            '并且，如果根的子女从左到右编号为k-1,k-2,...,0,子女i是子树Bi的根')
        print('推论19.2 在一棵包含n个结点的二项树中，任意结点的最大度数为lgn')
        print('19.1.2 二项堆')
        print('二项堆H由一组满足下面的二项堆性质的二项树组成')
        print('(1) H中的每个二项树遵循最小堆性质：',
            '结点的关键字大于或等于其父结点的关键字,我们说这种树是最小堆有序的')
        print('(2) 对任意非负整数k，在H中至多有一棵二项树的根具有度数k')
        print('在一棵最小堆有序的二项树中，其根包含了树中最小的关键字')
        print('在包含n个结点的二项堆H中，包含至多[lgn]+1棵二项树')
        print('这样，二项堆H包含至多[lgn]+1棵二项树')
        print('包含13个结点的二项堆H。13的二进制表示为1101，',
            '故H包含了最小堆有序二项树B3,B2和B0,它们分别有8,4,1个结点，即共有13个结点')
        print('二项堆的表示')
        print(' 在二项堆的每个结点中，都有一个关键字域及其其他依应用要求而定的卫星数据')
        print(' 另外，每个结点x还包含了指向其父结点的指针p[x],指向其最做孩子的指针child[x]')
        print(' 以及指向x的紧右兄弟的指针sibling[x].如果结点x是根，则p[x]=None')
        print(' 如果结点x没有子女，则child[x]=None,如果x是其父结点的最右孩子，则sibling[x]=None')
        print(' 如果结点x是根，则p[x]=None,如果结点x没有子女，',
            '则child[x]=None,如果x是其父结点的最右孩子，',
            '则sibling[x]=None,每个结点x都包含域degree[x],即x的子女个数')
        print('一个二项堆中的各二项树被组织成一个链表，我们称之为根表。')
        print('在遍历根表时，各根的度数是严格递增的')
        print('根据第二个二项堆的性质，在一个n结点的二项堆中各根的度数构成了{0,1,...,[lgn]}的一个子集')
        print('对根结点来说与非结点根来说，sibling域的含义是不同的，如果x为根，则x.sibling指向根表中下一个根')
        print('像通常一样，如果x为根表中最后一个根，则x.sibling=None')
        print('练习19.1-1 假设x为一个二项堆中，某棵二项树中的一个结点，并假定sibling[x]!=None')
        print(' 如果x不是根，x.sibling.degree比x.degree多1，',
            '如果x是个根，则x.sibling.degree比x.degree多至少1,因为需要知道二项堆的二项树组成结构')
        print('练习19.1-2 如果x是二项堆的某棵二项树的非根结点，x.p.degree比x.degree大至多O(n)')
        print('练习19.1-3 假设一棵二项树Bk中的结点标为二进制形式。考虑深度i处标为l的一个结点x，且设j=k-i.')
        print(' 证明：在x的二进制表示中共有j个1.恰好包含j个1的二进制k串共有多少？',
            '证明x的度数与l的二进制表示中，最右0的右边的1的个数相同')
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

class Chapter19_2:
    '''
    chpater19.2 note and function
    '''
    def buildheap(self):
        '''
        构造19.2-2的形式二项堆
        '''
        heap = bh.BinomialHeap()
        root1 = bh.BinomialHeapNode(25, 0)
        # 根结点
        heap.head = root1
        root2 = bh.BinomialHeapNode(12, 2)
        root3 = bh.BinomialHeapNode(6, 4)
        heap.head.sibling = root2
        root2.sibling = root3

        root2.child = bh.BinomialHeapNode(37, 1, root2)
        root2.child.sibling = bh.BinomialHeapNode(18, 0, root2)

        root2.child.child = bh.BinomialHeapNode(
            41, 0, root2.child)

        root3.child = bh.BinomialHeapNode(10, 3, root3)
        root3.child.sibling = bh.BinomialHeapNode(8, 2, root3)
        root3.child.sibling.sibling = bh.BinomialHeapNode(14, 1, root3)
        root3.child.sibling.sibling.sibling = bh.BinomialHeapNode(29, 0, root3)

        node = root3.child
        node.child = bh.BinomialHeapNode(16, 2, node)
        node.child.sibling = bh.BinomialHeapNode(28, 1, node)
        node.child.sibling.sibling = bh.BinomialHeapNode(13, 0, node)

        node = root3.child.sibling
        node.child = bh.BinomialHeapNode(11, 1, node)
        node.child.sibling = bh.BinomialHeapNode(17, 0, node)
        node.child.child = bh.BinomialHeapNode(27, 0, node.child)

        node = root3.child.sibling.sibling
        node.child = bh.BinomialHeapNode(38, 0, node)

        node = root3.child.child
        node.child = bh.BinomialHeapNode(26, 1, node)
        node.child.sibling = bh.BinomialHeapNode(23, 0, node)
        node.child.child = bh.BinomialHeapNode(42, 0, node.child)

        node = root3.child.child.sibling
        node.child = bh.BinomialHeapNode(77, 0, node)

        return heap

    def note(self):
        '''
        Summary
        ====
        Print chapter19.2 note

        Example
        ====
        ```python
        Chapter19_2().note()
        ```
        '''
        print('chapter19.2 note as follow')  
        print('19.2 对二项堆的操作')
        print('创建一个新二项堆')
        print(' 为了构造一个空的二项堆')
        print('寻找最小关键字')
        print(' 过程BINOMIAL-HEAP-MINIMUM返回一个指针，',
            '指向包含n个结点的二项堆H中具有最小关键字的结点',
            '这个实现假设没有一个关键字为无穷')
        print(' 因为一个二项堆是最小堆有序的，故最小关键字必在根结点中') 
        print(' 过程BINOMIAL-HEAP-MINIMUM检查所有的根(至多[lgn]+1),将当前最小者存于min中')
        print(' 而将指向当前最小者的指针存于y之中。BINOMIAL-HEAP-MINIMUM返回一个指向具有关键字1的结点的指针')
        print(' 因为至多要检查[lgn]+1个根，所以BINOMIAL-HEAP-MINIMUM的运行时间为O(lgn)')
        print('合并两个二项堆')
        print(' 合并两个二项堆的操作可用作后面大部分操作的一个子程序。')
        print(' 过程BINOMIAL-HEAP-UNION反复连接根结点的度数相同的各二项树')
        print(' LINK操作将以结点y为根的Bk-1树与以结点z为根的Bk-1树连接起来')
        print(' BINOMIAL-HEAP-UNION搓成合并H1和H2并返回结果堆，在合并过程中，同时也破坏了H1和H2的表示')
        print(' 还使用了辅助过程BINOMIAL-HEAP-MERGE,来讲H1和H2的根表合并成一个按度数的单调递增次序排列的链表')
        print('练习19.2-1 写出BINOMIAL-HEAP-MERGE的伪代码 代码已经给出')
        heap = bh.BinomialHeap()
        heap = heap.insertkey(1)
        heap = heap.insertkey(2)
        heap = heap.insertkey(3)
        print(heap.head)
        print('练习19.2-2 将关键字24的结点插入如图19-7d的二项树当中')
        heap = self.buildheap()
        print(heap.head)
        heap = heap.insertkey(24)
        print(heap.head)
        print(' 所得结果二项堆就是24变成了头结点，25变成24的子结点')
        heap = heap.deletekey(28)
        print('练习19.2-3 删除28关键字整个二项堆结构与原来很不相同')
        print('练习19.2-4 讨论使用如下循环不变式BINOMIAL-HEAP-UNION的正确性')
        print(' x指向下列之一的根')
        print(' 1.该度数下唯一的根')
        print(' 2.该度数下仅有两根中的第一个')
        print(' 3.该度数下仅有三个根中的第一或第二个')
        print('练习19.2-5 如果关键字的值可以是无穷，为什么过程BINOMIAL-HEAP-MINIMUM可能无法工作')
        print('练习19.2-6 假设无法表示出关键字负无穷')
        print(' 重写BINOMIAL-HEAP-DELETE过程，使之在这种情况下能正确地工作，运行时间仍然为O(lgn)')
        print('练习19.2-7 类似的')
        print(' 讨论二项堆上的插入与一个二进制数增值的关系')
        print(' 合并两个二项堆与将两个二进制数相加之间的关系')
        print('练习19.2-8 略')
        print('练习19.2-9 证明：如果将根表按度数排成严格递减序(而不是严格递增序)保存')
        print(' 仍可以在不改变渐进运行时间的前提下实现每一种二项堆操作')
        print('练习19.2-10 略')
        print('思考题19-1 2-3-4堆')
        print(' 2-3-4树，其中每个内结点(非根可能)有两个、三个或四个子女，且所有的叶结点的深度相同')
        print(' 2-3-4堆与2-3-4树有些不同之处。在2-3-4堆中，关键字仅存在于叶结点中，',
            '且每个叶结点x仅包含一个关键字于其x.key域中')
        print(' 另外，叶结点中的关键字之间没有什么特别的次序；亦即，从左至右看，各关键字可以排成任何次序')
        print(' 每个内结点x包含一个值x.small,它等于以x为根的子树的各叶结点中所存储的最小关键字')
        print(' 根r包含了一个r.height域,即树的高度。最后，2-3-4堆主要是在主存中的，故无需任何磁盘读写')
        print(' 2-3-4堆应该包含如下操作，其中每个操作的运行时间都为O(lgn)')
        print(' (a) MINIMUM,返回一个指向最小关键字的叶结点的指针')
        print(' (b) DECREASE-KEY,将某一给定叶结点x的关键字减小为一个给定的值k<=x.key')
        print(' (c) INSERT,插入具有关键字k的叶结点x')
        print(' (d) DELETE,删除一给定叶结点x')
        print(' (e) EXTRACT-MIN,抽取具有最小关键字的叶结点')
        print(' (f) UNION,合并两个2-3-4堆，返回一个2-3-4堆并破坏输入堆')
        print('思考题19-2 采用二项堆的最小生成树算法')
        print(' 第23章要介绍两个在无向图中寻找最小生成树的算法')
        print(' 可以利用二项堆来设计一个不同的最小生成树算法')
        print(' 请说明如何用二项堆来实现此算法，以便管理点集合边集。需要对二项堆的表示做改变嘛')
        # python src/chapter19/chapter19note.py
        # python3 src/chapter19/chapter19note.py

chapter19_1 = Chapter19_1()
chapter19_2 = Chapter19_2()

def printchapter19note():
    '''
    print chapter19 note.
    '''
    print('Run main : single chapter nineteen!')  
    chapter19_1.note()
    chapter19_2.note()

# python src/chapter19/chapter19note.py
# python3 src/chapter19/chapter19note.py
if __name__ == '__main__':  
    printchapter19note()
else:
    pass

```

```py

class BinomialHeapNode:
    '''
    二项堆结点
    '''
    def __init__(self, key=None, degree=None, p=None, 
        child = None, sibling = None):
        '''
        二项堆结点

        Args
        ===
        `p` : 父结点
        
        `key` : 关键字

        `degree` : 子结点的个数

        `child` : 子结点

        `sibling` : 二项堆同根的下一个兄弟

        '''
        self.p = p
        self.key = key
        self.degree = degree
        self.child = child
        self.sibling = sibling
    
    def __str__(self):
        '''
        str(self.key)
        '''
        return str(self.key)

class BinomialHeap:
    '''
    二项堆
    '''
    def __init__(self, head : BinomialHeapNode = None):
        '''
        二项堆

        Args
        ===
        `head` : 头结点

        '''
        self.head = head
        self.__findnode = None

    def minimum(self):
        '''
        求出指向包含n个结点的二项堆H中具有最小关键字的结点

        时间复杂度`O(lgn)`

        '''
        y = None
        x = self.head
        if x is not None:
            min = x.key 
        while x != None:
            if x.key < min:
                min = x.key
                y = x
            x = x.sibling
        return y

    @classmethod
    def link(self, y : BinomialHeapNode, z : BinomialHeapNode):
        '''
        将一结点`y`为根的Bk-1树与以结点`z`为根的Bk-1树连接起来
        也就是使`z`成为`y`的父结点，并且成为一棵Bk树

        时间复杂度`O(1)`

        Args
        ===
        `y` : 一个结点

        `z` : 另外一个结点
        '''
        y.p = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def insert(self, x : BinomialHeapNode):
        '''
        插入一个结点 时间复杂度`O(1) + O(lgn) = O(lgn)`
        '''
        h1 = BinomialHeap.make_heap()
        x.p = None
        x.child = None
        x.sibling = None
        x.degree = 0
        h1.head = x
        unionheap = BinomialHeap.union(self, h1)
        return unionheap

    def insertkey(self, key):
        '''
        插入一个结点 时间复杂度`O(1) + O(lgn) = O(lgn)`
        '''
        return self.insert(BinomialHeapNode(key))

    def deletekey(self, key):
        '''
        删除一个关键字为`key`的结点 时间复杂度`O(lgn)`
        '''
        node = self.findkey(key)
        if node is not None:
            return self.delete(node)
        return self

    def findkey(self, key):
        '''
        查找一个结点 时间复杂度`O(lgn)`
        '''
        self.__findkey(key, self.head)
        return self.__findnode

    def __findkey(self, key, node):
        '''
        查找一个结点
        '''
        if node is not None:
            if node.key == key:
                self.__findnode = node
            i = 0 
            nodefind = node.child
            while i < node.degree:
                self.__findkey(key, nodefind)
                nodefind = nodefind.sibling
                i += 1
            nodefind = node.sibling
            while nodefind is not None:
                self.__findkey(key, nodefind)
                nodefind = nodefind.sibling

    def extract_min(self):
        '''
        抽取最小关键字
        '''
        p = self.head
        x = None
        x_prev = None
        p_prev = None
        if p is None:
            return p
        x = p
        min = p.key
        p_prev = p
        p = p.sibling
        while p is not None:
            if p.key < min:
                x_prev = p_prev
                x = p
                min = p.key
            p_prev = p
            p = p.sibling
        if x == self.head:
            self.head = x.sibling
        elif x.sibling is None:
            x_prev.sibling = None
        else:
            x_prev.sibling = x.sibling
        child_x = x.child
        if child_x != None:
            h1 = BinomialHeap.make_heap()
            child_x.p = None
            h1.head = child_x
            p = child_x.sibling
            child_x.sibling = None
            while p is not None:
                p_prev = p
                p = p.sibling
                p_prev.sibling = h1.head
                h1.head = p_prev
                p_prev.p = None
            self = BinomialHeap.union(self, h1)         
        return self

    def decresekey(self, x : BinomialHeapNode, key):
        '''
        减小结点的关键字的值，调整该结点在相应二项树中的位置
        '''
        if x.key < key:
            return 
        x.key = key
        y = x
        p = x.p
        while p is not None and y.key < p.key:
            y.key = p.key
            p.key = key
            y = p
            p = y.p

    def delete(self, x : BinomialHeapNode):
        '''
        删除一个关键字 时间复杂度`O(lgn)`
        '''
        self.decresekey(x, -2147483648)
        return self.extract_min()

    @classmethod
    def make_heap(self):
        '''
        创建一个新的二项堆
        '''
        heap = BinomialHeap()
        return heap

    @classmethod
    def merge(self, h1, h2):
        '''
        合并两个二项堆h1和h2
        '''
        firstNode = None
        p = None
        p1 = h1.head
        p2 = h2.head
        if p1 is None or p2 is None:
            if p1 is None:
                firstNode = p2
            else:
                firstNode = p1
            return firstNode
        if p1.degree < p2.degree:
            firstNode = p1
            p = firstNode
            p1 = p1.sibling
        else:
            firstNode = p2
            p = firstNode
            p2 = p2.sibling
        while p1 is not None and p2 is not None:
            if p1.degree < p2.degree:
                p.sibling = p1
                p = p1
                p1 = p1.sibling
            else:
                p.sibling = p2
                p = p2
                p2 = p2.sibling
        if p1 is not None:
            p.sibling = p1
        else:
            p.sibling = p2
        return firstNode
    
    @classmethod
    def union(self, h1, h2):
        '''
        两个堆合并 时间复杂度:`O(lgn)`
        '''
        h = BinomialHeap.make_heap()
        h.head = BinomialHeap.merge(h1, h2)
        del h1
        del h2
        if h.head is None:
            return h
        prev = None
        x = h.head
        next = x.sibling
        while next is not None:
            if x.degree != next.degree or (next.sibling is not None and x.degree == next.sibling.degree):
                prev = x
                x = next
            elif x.key <= next.key:
                x.sibling = next.sibling
                h.link(next, x)
            else:
                if prev is None:
                    h.head = next
                else:
                    prev.sibling = next
                h.link(x, next)
            next = x.sibling
        return h
                
def test():
    '''
    test
    '''
    print('BinomialHeapNode and BinomialHeap test')
    heap = BinomialHeap.make_heap()

    node = BinomialHeapNode(5)
    heap = heap.insert(node)
    heap = heap.insertkey(8)
    heap = heap.insertkey(2)
    heap = heap.insertkey(7)
    heap = heap.insertkey(6)
    heap = heap.insertkey(9)
    heap = heap.insertkey(4)
    heap = heap.extract_min()
    print(heap.minimum())
    heap = heap.delete(node)
    if heap.head is not None:
        print(heap.head)
    heap = heap.extract_min()
    heap = heap.deletekey(9)
    if heap.head is not None:
        print(heap.head)

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter20/chapter20note.py
# python3 src/chapter20/chapter20note.py
'''

Class Chapter20_1

Class Chapter20_2

Class Chapter20_3

Class Chapter20_4

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

if __name__ == '__main__':
    import fibonacciheap as fh
else:
    from . import fibonacciheap as fh

class Chapter20_1:
    '''
    chpater20.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter20.1 note

        Example
        ====
        ```python
        Chapter20_1().note()
        ```
        '''
        print('chapter20.1 note as follow')  
        print('第20章 斐波那契堆')
        print('二项堆可以在时间O(lgn)的最坏情况时间支持可合并堆操作INSERT,MINIMUM,',
            'EXTRACT-MIN和UNION以及操作DECREASE-KEY和DELETE')
        # !斐波那契堆不涉及删除元素的可合并堆操作仅需要O(1)的平摊时间
        print('波那契堆不涉及删除元素的可合并堆操作仅需要O(1)的平摊时间,这是斐波那契堆的好处')
        print('从理论上来看，当相对于其他操作的数目来说，EXTRACT-MIN与DELETE操作的数目较小时,斐波那契堆是很理想的')
        print(' 例如某些图问题的算法对每条边都调用一次DECRESE-KEY。对有许多边的稠密图来说，每一次DECREASE-KEY调用O(1)平摊时间加起来')
        print(' 就是对二叉或二项堆的Θ(lgn)最坏情况时间的一个很大改善')
        print(' 比如解决诸如最小生成树和寻找单源最短路径等问题的快速算法都要用到斐波那契堆')
        print('但是，从实际上看，对大多数应用来说，由于斐波那契堆的常数因子以及程序设计上的复杂性',
            '使得它不如通常的二叉(或k叉)堆合适')
        print('因此，斐波那契堆主要是具有理论上的意义')
        print('如果能设计出一种与斐波那契堆有相同的平摊时间界但又简单得多的数据结构，',
            '那么它就会有很大的实用价值了')
        # !和二项堆一样，斐波那契堆由一组树构成
        print('和二项堆一样，斐波那契堆由一组树构成，实际上，这种堆松散地基于二项堆')
        print('如果不对斐波那契堆做任何DECREASE-KEY或DELETE操作，则堆中的每棵树就和二项树一样')
        print('两种堆相比，斐波那契堆的结构比二项堆更松散一些，可以改善渐进时间界。',
            '对结构的维护工作可被延迟到方便再做')
        # !斐波那契堆也是以平摊分析为指导思想来设计数据结构的很好的例子(可以利用势能方法)
        print('斐波那契堆也是以平摊分析为指导思想来设计数据结构的很好的例子(可以利用势能方法)')
        # !和二项堆一样，斐波那契堆不能有效地支持SEARCH操作
        print('和二项堆一样，斐波那契堆不能有效地支持SEARCH操作')
        print('20.1 斐波那契堆的结构')
        # !与二项堆一样，斐波那契堆是由一组最小堆有序树构成，但堆中的树并不一定是二项树
        print('与二项堆一样，斐波那契堆是由一组最小堆有序树构成，但堆中的树并不一定是二项树')
        # !与二项堆中树都是有序的不同，斐波那契堆中的树都是有根而无序的
        print('与二项堆中树都是有序的不同，斐波那契堆中的树都是有根而无序的')
        print('每个结点x包含一个指向其父结点的指针p[x],以及一个指向其任一子女的指针child[x],',
            'x所有的子女被链接成一个环形双链表,称为x的子女表。',
            '子女表的每个孩子y有指针left[y]和right[y]分别指向其左，右兄弟')
        print('如果y结点是独子，则left[y]=right[y]=y。各兄弟在子女表中出现的次序是任意的')
        # !在斐波那契堆中采用环形双链表
        print('在斐波那契堆中采用环形双链表有两个好处。',
            '首先可以在O(1)时间内将某结点从环形双链表中去掉')
        print(' 其次，给定两个这样的表，可以在O(1)时间内将它们连接为一个环形双链表')
        print('另外一个结点中的另外两个域也很有用。结点x的子女表中子女的个数存储于degree[x]')
        print('布尔值域mark[x],指示自从x上一次成为另一个结点子女以来，它是否失掉了一个孩子')
        print('新创建的结点是没有标记的，且当结点x成为另一个结点的孩子时也是没有标记的')
        print('DECREASE-KEY操作，才会置所有的mark域为FALSE')
        print('对于给定的斐波那契堆H,可以通过指向包含最小关键字的树根的指针min[H]来访问')
        print(' 这个结点被称为斐波那契堆中的最小结点。如果一个斐波那契堆H是空的，则min[H]=None')
        print('在一个斐波那契堆中，所有树的根都通过用其left和right指针链接成一个环形的双链表，称为该堆的根表')
        print('在根表中各树的顺序可以是任意的')
        print('斐波那契堆H中目前所包含的结点个数为n[H]')
        print('势函数')
        print(' 对一个给定的斐波那契堆H，用t(H)来表示H的根表中树的棵数，用m(H)来表示H中有标记结点的个数')
        print(' 斐波那契堆H的势定义为Φ(H)=t(H)+2m(H)')
        print(' 一组斐波那契堆的势为各成分斐波那契堆的势之和。')
        print(' 假定一个单位的势可以支付常数量的工作，此处该常数足够大，',
            '可以覆盖我们可能遇到的任何常数时间的工作')
        print(' 假定斐波那契堆应用在开始时，都没有堆。于是，初始的势就为0，且根据方程，势始终是非负的')
        print(' 对某一操作序列来说，其总的平摊代价的一个上界也就是这个操作序列总的实际代价的一个上界')
        print('最大度数')
        print(' 在包含n个结点的斐波那契堆中，结点的最大度数有一个已知的上界D(n)')
        print(' 当斐波那契堆仅支持可合并堆操作时,D(n)<=[lgn]')
        print(' 当斐波那契堆还需支持DECREASE-KEY和DELETE操作时,D(n)=O(lgn)')
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

class Chapter20_2:
    '''
    chpater20.2 note and function
    '''
    def putxkeydownykey(self, heap : fh.FibonacciHeap, xkey, ykey):
        '''
        构造20.3的斐波那契堆
        '''
        ynode = heap.search(ykey)
        root = ynode
        node = fh.make_fib_node(xkey)
        fh.link(node, root)
        heap.keynum += 1

    def buildheap(self):
        '''
        构造20.3的斐波那契堆
        '''
        heap = fh.make_fib_heap()
        heap.insertkey(7)
        heap.insertkey(18)
        heap.insertkey(38)

        self.putxkeydownykey(heap, 24, 7)
        self.putxkeydownykey(heap, 17, 7)
        self.putxkeydownykey(heap, 23, 7)

        self.putxkeydownykey(heap, 26, 24)
        self.putxkeydownykey(heap, 46, 24)
        self.putxkeydownykey(heap, 35, 26)

        self.putxkeydownykey(heap, 30, 17)

        self.putxkeydownykey(heap, 21, 18)
        self.putxkeydownykey(heap, 39, 18)
        self.putxkeydownykey(heap, 52, 21)

        self.putxkeydownykey(heap, 41, 38)

        return heap

    def note(self):
        '''
        Summary
        ====
        Print chapter20.2 note

        Example
        ====
        ```python
        Chapter20_2().note()
        ```
        '''
        print('chapter20.2 note as follow')  
        print('20.2 可合并堆的操作')
        print('介绍斐波那契堆所实现的各种可合并堆操作。',
            '如果仅需要支持MAKE-HEAP,INSERT,MINIMUM,EXTRACT-MI和UNION操作')
        print('则每个斐波那契堆就只是一组\"无序的\"的二项树')
        print('无序的二项树和二项树一样，也是递归定义的')
        print('无序的二项树U0包含一个结点，一棵无序的二项树Uk包含两颗无序的二项树Uk-1')
        # !如果一个有n个结点的斐波那契堆由一组无序二项树构成，则D(n)=lgn
        print('可能要花Ω(lgn)时间向一个二项堆中插入一个结点或合并两个二项堆。',
            '当向斐波那契堆中插入新结点或者合并两个斐波那契堆时，并不去合并树')
        print(' 而是将这个工作留给EXTRACT-MIN操作，那是就真正需要找出新的最小结点了')
        print('斐波那契堆插入一个结点')
        print(' 与BINOMIAL-HEAP-INSERT过程不同，FIB-HEAP-INSERT并不对斐波那契堆中的树进行合并')
        print(' 如果连续执行了k次FIB-HEAP-INSERT操作，则k棵单结点的树被加到了根表中')
        print(' 为了确定FIB-HEAP-INSERT的平摊代价，设H为输入的斐波那契堆，H‘为结果斐波那契堆')
        print(' 于是，t(H‘)=t(H)+1,m(H‘)=m(H),且势的增加为1')
        print(' 因为实际的代价为O(1)，故平摊的代价为O(1)+1=O(1)')
        print('寻找最小结点')
        print(' 在一个斐波那契堆H中,最小的结点由指针min[H],故可以在O(1)实际时间内找到最小结点')
        print(' 又因为H的势没有变化，所以这个操作的平摊代价就等于其O(1)的实际代价')
        print('合并两个斐波那契堆')
        print(' 合并斐波那契堆H1和H2，同时破坏H1和H2，仅仅简单地将H1和H2的两根表并置,然后确定一个新的最小结点')
        print(' 势的改变为0,FIB-HEAP-UNION的平摊代价与其O(1)的实际代价相等')
        print('抽取最小结点')
        print(' 斐波那契堆FIB-HEAP-EXTRACT-MIN的操作是最复杂的。')
        print(' 先前对根表中的树进行合并这项工作是被推后的，那么到了这儿最终必须由这个操作完成')
        print(' 假定从链表中删除一个结点时，仍在表中的指针被更新，而被抽取结点的指针则无变化。该过程还用到辅助过程CONSOLIDATE')
        print(' 在抽取最小结点之前的势为t(H)+2m(H),而在此之后的势至多为(D(n)+1)+2m(H)')
        print(' 因为在该操作之后至多留下D(n)+1个根，且操作中没有任何结点被加标记，所以总的平摊代价至多为O(D(n))')
        print(' 这是因为可以通过扩大势的单位来支配O(t(H))中隐藏的常数。')
        print(' 从直觉上看，执行每一次链接的代价是由势的减少来支付的，',
            '而势的减少又是由于链接操作使根的数目减少1而引起的')
        print(' 若D(n)=O(lgn),所以抽取最小结点的平摊代价为O(lgn)')
        print('练习20.2-1 给出图20-3中所示的斐波那契堆调用FIB-HEAP-EXTRACT-MIN后得到的斐波那契堆')
        heap = self.buildheap()
        print('remove before')
        heap.print()
        heap.extractmin()
        print('remove after')
        heap.print()
        print('练习20.2-2 证明：引理19.1(二项树的性质)对无序二项树也成立')
        print('练习20.2-3 证明：如果仅需支持可合并堆操作，',
            '则在包含n个结点的斐波那契堆中结点的最大度数D(n)至多为[lgn]')
        print('练习20.2-4 McGee教授设计了一种新的基于斐波那契堆的数据结构')
        print(' McGee堆与斐波那契堆具有相同的结构，也支持可合并堆操作。',
            '各操作的实现与斐波那契堆中的相同,不同的是插入和合并在最后的步骤中做合并调整')
        print('练习20.2-5 论证：如果关键字的唯一操作是比较两个关键字，',
            '则并非所有的可合并堆操作都有O(1)的平摊时间')
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

class Chapter20_3:
    '''
    chpater20.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter20.3 note

        Example
        ====
        ```python
        Chapter20_3().note()
        ```
        '''
        print('chapter20.3 note as follow')
        print('20.3 减小一个关键字于删除一个结点')
        print('减小斐波那契堆中某结点的关键字，平摊时间为O(1)')
        print('从包含n个结点的斐波那契堆中删除一个结点，平摊时间O(D(n))')
        print('这些操作不包吃斐波那契堆中的所有树都是无序二项树的性质')
        print('因而可以用O(lgn)来限界最大度数D(n)')
        print('FIB-HEAP-EXTRACT-MIN和FIB-HEAP-DELETE的平摊运行时间为O(lgn)')
        print('练习20.3-1 假设一个斐波那契堆中某个根x是有标记的。')
        print(' 请解释x是如何成为有标记的根')
        print(' 另说明x有无标记对分析来说没有影响，',
            '即使他不是先被链接到另一个结点，然后又失去一个子结点的根')
        print('练习20.3-2 使用聚焦分析来证明FIB-HEAP-DECREASE-KEY的平摊时间O(1)是每个操作的平均代价')
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

class Chapter20_4:
    '''
    chpater20.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter20.4 note

        Example
        ====
        ```python
        Chapter20_4().note()
        ```
        '''
        print('chapter20.4 note as follow')
        print('20.4 最大度数的界')
        print('为了证明FIB-HEAP-EXTRACT-MIN和FIB-HEAP-DELETE的平摊时间为O(lgn)')
        print('必须首先证明，在包含n个结点的斐波那契堆中，任意结点的度数的上界D(n)为O(lgn)')
        print('引理20.1 设x为斐波那契堆中任一结点，且假设degree[x]=k，',
            '设y1,y2,...,yk表示按与x链接的次序排列的x的子女，从最早的到最迟的，则对i=2,3,...,k',
            '有degree[y1]>=0和degree[yi]>=i-2')
        print('引理20.2 对所有整数k>=0,F(k+2)=1+∑F(i)')
        print('引理20.3 设x为斐波那契的任一结点，且k=degree[x],那么size(x)>=Fk+2>=Φ^k,其中Φ=(1+sqrt(5))/2')
        print('推论20.4 在一个包含n个结点的斐波那契堆中，结点的最大度数D(n)为O(lgn)')
        print('练习20.4-1 Pinocchio教授声称，包含n个结点的斐波那契堆的高度为O(lgn)')
        print(' 请证明他是错的，即对于任意的正整数n，给出一个斐波那契堆操作序列，',
            '它创建一个仅包含一棵树的堆,该树是n个结点的线性链')
        print('练习20.4-2 假设将级联切断规则加以推广，使得当某个结点x失去其第k个孩子时')
        print(' 就将其与父结点的联系切断，此处k为常整数。k取什么值时，有D(n)=O(lgn)')
        print('思考题20-1 删除的另一种实现')
        print('思考题20-2 其他斐波那契堆的操作')
        print(' 增强斐波那契堆H，使之支持两种新的操作，同时，还不改变其他斐波那契堆操作的平摊运行时间')
        print('FIB-HEAP-CHANGE-KEY(H, x. k) 将结点x的关键字改变为k')
        print('FIB-HEAP-PRUNE(H, r) 将min(r, n[H])个结点从H中删除')
        # python src/chapter20/chapter20note.py
        # python3 src/chapter20/chapter20note.py

chapter20_1 = Chapter20_1()
chapter20_2 = Chapter20_2()
chapter20_3 = Chapter20_3()
chapter20_4 = Chapter20_4()

def printchapter20note():
    '''
    print chapter20 note.
    '''
    print('Run main : single chapter twenty!')  
    chapter20_1.note()
    chapter20_2.note()
    chapter20_3.note()
    chapter20_4.note()

# python src/chapter20/chapter20note.py
# python3 src/chapter20/chapter20note.py
if __name__ == '__main__':  
    printchapter20note()
else:
    pass

```

```py

import math as _math

class FibonacciHeapNode:
    '''
    斐波那契堆结点
    '''
    def __init__(self, key = None, degree = None, p = None, child = None, \
        left = None, right = None, mark = None):
        '''
        斐波那契堆结点

        Args
        ===
        `key` : 关键字值

        `degree` : 子结点的个数

        `p` : 父结点

        `child` : 任意一个子结点

        `left` : 左兄弟结点

        `right` : 右兄弟结点

        `mark` : 自从x上一次成为另一个结点子女以来，它是否失掉了一个孩子

        '''
        self.key = key
        self.degree = degree
        self.p = p
        self.child = child
        self.left = left
        self.right = right
        self.mark = mark

    def __str__(self):
        return str({"key" : self.key, \
            "degree" : self.degree})

def make_fib_node(key):
    '''
    创建斐波那契堆的结点
    Example
    ===
    ```python
    import fibonacciheap as fh
    >>> node = fh.make_fib_node()
    ```
    '''
    node = FibonacciHeapNode()
    node.key = key
    node.degree = 0
    node.left = node
    node.right = node
    node.p = None
    node.child = None
    return node

def make_fib_heap():
    '''
    创建一个新的空的斐波那契堆

    平摊代价和实际代价都为`O(1)`

    Example
    ===
    ```python
    import fibonacciheap as fh
    heap = fh.make_fib_heap()
    ```
    '''
    heap = FibonacciHeap()
    heap.keynum = 0
    heap.maxdegree = 0
    heap.min = None
    heap.cons = None
    return heap

def node_add(node: FibonacciHeapNode, root: FibonacciHeapNode):
    '''
    将单个结点`node``加入链表`root`之前

    此处`node`是单个结点，`root`是双向链表

    '''
    if node is None or root is None:
        return
    node.left = root.left
    root.left.right = node
    node.right = root
    root.left = node

def node_remove(node: FibonacciHeapNode):
    '''
    将单个结点`node`从双链表中移除
    '''
    node.left.right = node.right
    node.right.left = node.left

def node_cat(a: FibonacciHeapNode, b: FibonacciHeapNode):
    '''
    将双向链表`b`链接到双向链表`a`的后面
    '''
    tmp = a.right
    a.right = b.right
    b.right.left = a
    b.right = tmp
    tmp.left = b

def link(node: FibonacciHeapNode, root: FibonacciHeapNode):
    '''
    将`node`链接到`root`根结点
    '''
    FibonacciHeap.node_remove(node)
    if root is None:
        return
    if root.child == None:
        root.child = node
    else:
        FibonacciHeap.node_add(node, root.child)
    node.p = root
    root.degree += 1
    node.mark = False

class FibonacciHeap:
    '''
    斐波那契堆 不涉及删除元素的可合并堆操作仅需要`O(1)`的平摊时间,这是斐波那契堆的优点
    '''
    def __init__(self):
        '''
        斐波那契堆 不涉及删除元素的可合并堆操作仅需要`O(1)`的平摊时间,这是斐波那契堆的优点
        '''
        self.min = None
        self.cons = []
        self.keynum = 0
        self.maxdegree = 0
        
    @classmethod
    def make_fib_heap(self):
        '''
        创建一个新的空的斐波那契堆

        平摊代价和实际代价都为`O(1)`

        Example
        ===
        ```python
        heap = FibonacciHeap.make_fib_heap()
        ```
        '''
        heap = FibonacciHeap()
        heap.keynum = 0
        heap.maxdegree = 0
        heap.min = None
        heap.cons = None
        return heap
    
    @classmethod
    def make_fib_node(self, key):
        '''
        创建斐波那契堆的结点
        Example
        ===
        ```python
        node = FibonacciHeap.make_fib_node()
        ```
        '''
        node = FibonacciHeapNode()
        node.key = key
        node.degree = 0
        node.left = node
        node.right = node
        node.p = None
        node.child = None
        return node

    @classmethod
    def node_add(self, node : FibonacciHeapNode, root : FibonacciHeapNode):
        '''
        将单个结点`node``加入链表`root`之前

        此处`node`是单个结点，`root`是双向链表

        '''
        if node is None or root is None:
            return
        node.left = root.left
        root.left.right = node
        node.right = root
        root.left = node

    @classmethod
    def node_remove(self, node: FibonacciHeapNode):
        '''
        将单个结点`node`从双链表中移除
        '''
        node.left.right = node.right
        node.right.left = node.left

    @classmethod
    def node_cat(self, a : FibonacciHeapNode, b : FibonacciHeapNode):
        '''
        将双向链表`b`链接到双向链表`a`的后面
        '''
        tmp = a.right
        a.right = b.right
        b.right.left = a
        b.right = tmp
        tmp.left = b

    @classmethod
    def union(self, h1, h2):
        '''
        将`h1`和`h2`合并成一个堆，并返回合并后的堆
        '''
        tmp = None
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        if h2.maxdegree > h1.maxdegree:
            tmp = h1
            h1 = h2
            h2 = tmp

        if h1.min is None:
            h1.min = h2.min
            h1.keynum = h2.keynum
            del h2.cons
            del h2
        elif h2.min is None:
            del h2.cons
            del h2
        else:
            FibonacciHeap.node_cat(h1.min, h2.min)
            if h1.min.key > h2.min.key:
                h1.min = h2.min
            h1.keynum += h2.keynum
            del h2.cons
            del h2
        return h1

    def insertkey(self, key):
        '''
        插入一个关键字为`key`结点`x`
        '''
        node = FibonacciHeap.make_fib_node(key)
        self.insert(node)

    def insert(self, node : FibonacciHeapNode):
        '''
        插入一个结点`node`到斐波那契堆heap
        '''
        if self.keynum == 0:
            self.min = node
        else:
            FibonacciHeap.node_add(node, self.min)
            if node.key < self.min.key:
                self.min = node
        self.keynum += 1
    
    def unionwith(self, h2):
        '''
        将`self`和`h2`合并成一个堆，并返回合并后的堆
        '''
        tmp = None
        if self is None:
            return h2
        if h2 is None:
            return self

        if h2.maxdegree > self.maxdegree:
            tmp = self
            self = h2
            h2 = tmp

        if self.min is None:
            self.min = h2.min
            self.keynum = h2.keynum
            del h2.cons
            del h2
        elif h2.min is None:
            del h2.cons
            del h2
        else:
            FibonacciHeap.node_cat(self.min, h2.min)
            if self.min.key > h2.min.key:
                self.min = h2.min
            self.keynum += h2.keynum
            del h2.cons
            del h2
        return self

    def removemin(self):
        '''
        将堆的最小节点从根链表中移除

        意味着将最小结点所属的树移除
        '''
        min = self.min
        if self.min == min.right:
            self.min = None
        else:
            FibonacciHeap.node_remove(min)
            self.min = min.right
        min.right = min
        min.left = min.right
        return min

    def link(self, node: FibonacciHeapNode, root: FibonacciHeapNode):
        '''
        将`node`链接到`root`根结点
        '''
        FibonacciHeap.node_remove(node)
        if root.child == None:
            root.child = node
        else:
            FibonacciHeap.node_add(node, root.child)
        node.p = root
        root.degree += 1
        node.mark = False

    def cons_make(self):
        '''
        创建CONSOLIDATE过程所需空间
        '''
        old = self.maxdegree
        self.maxdegree = int(_math.log2(self.keynum) + 1)
        if old >= self.maxdegree:
            return
        self.cons = []
        for i in range(self.maxdegree + 1):
            self.cons.append(None)

    def consolidate(self):
        '''
        合并斐波那契堆的根链表中左右相同的度数
        '''
        i, d, D = 0, 0, 0
        x, y, tmp = None, None, None
        self.cons_make()
        D = self.maxdegree + 1
        for i in range(D):
            self.cons[i] = None
        while self.min is not None:
            x = self.removemin()
            d = x.degree
            while self.cons[d] is not None:
                y = self.cons[d]
                if x.key > y.key:
                    x, y = y, x
                self.link(y, x)
                self.cons[d] = None
                d += 1
            self.cons[d] = x
        self.min = None
        for i in range(D):
            if self.cons[i] is not None:
                if self.min is None:
                    self.min = self.cons[i]
                else:
                    FibonacciHeap.node_add(self.cons[i], self.min)
                    if self.cons[i].key < self.min.key:
                        self.min = self.cons[i]
        
    def __extractmin(self):
        '''
        移除最小节点，并返回最小结点
        '''
        if self is None or self.min is None:
            return None
        child = None
        min = self.min
        while min.child is not None:
            child = min.child
            FibonacciHeap.node_remove(child)
            if child.right == child:
                min.child = None
            else:
                min.child = child.right
            FibonacciHeap.node_add(child, self.min)
            child.p = None
        
        FibonacciHeap.node_remove(min)
        if min.right == min:
            self.min = None
        else:
            self.min = min.right
            self.consolidate()
        self.keynum -= 1
        return min
    
    def extractmin(self):
        '''
        移除最小节点，并返回最小结点 平摊运行时间为`O(lgn)`
        '''
        if self.min is None:
            return
        node = self.__extractmin()
        if node is not None:
            del node

    def get_min_key(self):
        '''
        获取最小结点关键字值
        '''
        if self.min is None:
            return None
        return self.min.key

    @classmethod
    def renew_degree(self, parent : FibonacciHeapNode, degree : int):
        '''
        修改度数
        '''
        parent.degree -= degree
        if parent.p is not None:
            self.renew_degree(parent.p, degree)
    
    def cut(self, node : FibonacciHeapNode, parent : FibonacciHeapNode):
        '''
        将`node`从父结点`parent`的子链接中剥离出来,并使`node`成为
        '''
        FibonacciHeap.node_remove(node)
        FibonacciHeap.renew_degree(parent, node.degree)
        if node == node.right:
            parent.child = None
        else:
            parent.child = node.right
        node.p = None
        node.left = node
        node.right = node
        node.mark = False
        FibonacciHeap.node_add(node, self.min)

    def cascading_cut(self, node : FibonacciHeapNode):
        '''
        对节点`node``进行级联剪切

        级联剪切：如果减小后的结点破坏了最小堆的性质，则把它切下来
        (即从所在双向链表中删除，并将其插入到由最小树根结点形成的双向链表中)，
        然后再从被切结点的父结点到所在树根结点递归执行级联裁剪
        '''
        parent = node.p
        if parent is not None:
            return
        if node.mark == False:
            node.mark = True
        else:
            self.cut(node, parent)
            self.cascading_cut(parent)
    
    def decrease(self, node : FibonacciHeapNode, key):
        '''
        将斐波那契堆heap中结点`node`的值减少为`key`

        平摊运行时间为`O(lgn)`

        '''
        if self.min is None or node is None:
            return
        assert key < node.key
        node.key = key
        parent = node.p
        if parent is not None and node.key < parent.key:
            self.cut(node, parent)
            self.cascading_cut(parent)
        if node.key < self.min.key:
            self.min = node

    def increase(self, node: FibonacciHeapNode, key):
        '''
        将斐波那契堆heap中结点`node`的值增加为`key`
        '''
        if self.min is None or node is None:
            return
        assert key > node.key
        while node.child is not None:
            child = node.child
            FibonacciHeap.node_remove(child)
            if child.right == child:
                node.child = None
            else:
                node.child = child.right
            FibonacciHeap.node_add(child, self.min)
            child.p = None
        node.degree = 0
        node.key = key
        parent = node.p
        if parent is not None:
            self.cut(node, parent)
            self.cascading_cut(parent)
        elif self.min == node:
            right = node.right
            while right is not node:
                if node.key > right.key:
                    self.min = right
                right = right.right

    def update(self, node: FibonacciHeapNode, key):
        '''
        更新二项堆heap的结点`node`的键值为`key`
        '''
        if key < node.key:
            self.decrease(node, key)
        elif key > node.key:
            self.increase(node, key)
        else:
            pass

    def updatekey(self, oldkey, newkey):
        '''
        更新二项堆heap的结点`node`的键值为`key`
        '''
        pass
    
    @classmethod
    def search_fromroot(self, root : FibonacciHeapNode, key):
        '''
        在最小堆`root`中查找键值为`key`的结点
        '''
        t = root
        p = None
        if root is None:
            return root
        while t != root.left:
            if t.key == key:
                p = t
                break
            else:
                p = self.search_fromroot(t.child, key)
                if p is not None:
                    break
                t = t.right
        else:
            if t.key == key:
                p = t
            else:
                p = self.search_fromroot(t.child, key)
        return p

    def search(self, key):
        '''
        在斐波那契堆heap中查找键值为`key`的节点
        '''
        if self.min is None:
            return None
        return self.search_fromroot(self.min, key)

    def delete(self, node : FibonacciHeapNode):
        '''
        删除结点`node`
        '''
        if self.min is None:
            return
        min = self.min
        self.decrease(node, min.key - 1)
        self.extractmin()
        del node
    
    def deletekey(self, key):
        '''
        删除关键字为`key`的结点`node`
        '''
        if self.min is None:
            return
        node = self.search(key)
        if node is None:
            return
        self.delete(node)

    @classmethod
    def destroynode(self, node : FibonacciHeapNode):
        '''
        销毁斐波那契堆
        '''
        if node is None:
            return
        start = node
        while node != start.left:
            self.destroy(node.child)
            node = node.right
            del node.left
        else:
            self.destroy(node.child)
            node = node.right
            del node.left

    def destroy(self):
        '''
        销毁斐波那契堆
        '''
        FibonacciHeap.destroynode(self.min)
        del self.cons

    @classmethod
    def print_node(self, node : FibonacciHeapNode, prev : FibonacciHeapNode, direction : int):
        '''
        打印"斐波那契堆"结点

        Args
        ===
        `node` : 当前结点

        `prev` : 当前结点的前一个结点(父结点或者兄弟结点)

        `direction` : 1表示当前结点是一个左孩子；2表示当前结点是一个兄弟结点

        '''
        start = node
        if node is None:
            return
        while node != start.left:
            if direction == 1:
                print('%8d(%d) is %2d\'s child' % (node.key, node.degree, prev.key))
            elif direction == 2:
                print('%8d(%d) is %2d\'s child' % (node.key, node.degree, prev.key))
            if node.child is not None:
                self.print_node(node.child, node, 1)
            prev = node
            node = node.right
            direction = 2
        else:
            if direction == 1:
                print('{}({}) is {}\'s child'.format(
                    node.key, node.degree, prev.key))
            elif direction == 2:
                print('{}({}) is {}\'s next'.format(node.key, node.degree, prev.key))
            if node.child is not None:
                self.print_node(node.child, node, 1)
            prev = node
            node = node.right
            direction = 2
         
    def print(self):
        '''
        打印"斐波那契堆"结点
        '''
        if self.min is None:
            return
        p = self.min
        i = 0
        while p != self.min.left:
            i += 1
            print("%2d. %4d(%d) is root " % (i, p.key, p.degree))
            FibonacciHeap.print_node(p.child, p, 1)
            p = p.right
        else:
            i += 1
            print("%2d. %4d(%d) is root " % (i, p.key, p.degree))
            FibonacciHeap.print_node(p.child, p, 1)
            p = p.right

def test():
    '''
    test
    '''
    print('FibonacciHeappNode and FibonacciHeap test')
    heap = FibonacciHeap.make_fib_heap()
    heap.insertkey(3)
    heap.insertkey(1)
    heap.insertkey(2)
    heap.insertkey(8)
    print(heap.min)
    heap.extractmin()
    print(heap.min)
    heapwillunion = FibonacciHeap.make_fib_heap()
    heapwillunion.insertkey(4)
    heapwillunion.insertkey(6)
    heapwillunion.insertkey(1)
    heapwillunion.insertkey(7)
    heap.unionwith(heapwillunion)
    print(heap.min)
    heap.deletekey(7)
    print(heap.min)
    heap.print()

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter21/chapter21note.py
# python3 src/chapter21/chapter21note.py
'''

Class Chapter21_1

Class Chapter21_2

Class Chapter21_3

Class Chapter21_4

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

if __name__ == '__main__':
    import notintersectset as _nset
else:
    from . import notintersectset as _nset

class Chapter21_1:
    '''
    chpater21.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.1 note

        Example
        ====
        ```python
        Chapter21_1().note()
        ```
        '''
        print('chapter21.1 note as follow')  
        print('第21章 用于不相交集合的数据结构')
        print('某些应用中，要将n个不同的元素分成一组不想交的集合')
        print('不相交集合上有两个重要的操作，即找出给定的元素所属的集合和合并两个集合')
        print('为了使某种数据结构能够支持这两种操作，就需要对该数据结构进行维护；本章讨论各种维护方法')
        print('21.1描述不相交集合数据结构所支持的各种操作，并给出这种数据结构的一个简单应用')
        print('21.2介绍不想交集合的一种简单链表实现')
        print('21.3采用有根树的表示方法的运行时间在实践上来说是线性的，但从理论上来说是超线性的')
        print('21.4定义并讨论一种增长极快的函数以及增长极为缓慢的逆函数')
        print(' 在基于树的实现中，各操作的运行时间中都出现了该反函数。',
            '然后，再利用平摊分析方法，证明运行时间的一个上界是超线性的')
        print('21.1 不相交集合上的操作')
        print('不相交集合数据结构保持一组不相交的动态集合S={S1,S2,...,Sk}')
        print('每个集合通过一个代表来识别，代表即集合中的某个成员')
        print('集合中的每一个元素是由一个对象表示的，设x表示一个对象，希望支持以下操作')
        print('MAKE-SET(x)：其唯一成员(因而其代表)就是x。因为各集合是不相交的，故要求x没有在其他集合出现过')
        print('UNION(x, y)：将包含x和y的动态集合(比如说Sx和Sy)合并为一个新的集合(并集)')
        print('FIND-SET(x)：返回一个指针，指向包含x的(唯一)集合的代表')
        print('不相交集合数据结构的一个应用')
        # !不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数
        print(' 不相交集合数据结构有多种应用,其中之一是用于确定一个无向图中连通子图的个数')
        g = _nset.UndirectedGraph()
        g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        g.edges = [('d', 'i'), ('f', 'k'), ('g', 'i'),
                   ('b', 'g'), ('a', 'h'), ('i', 'j'), 
                   ('d', 'k'), ('b', 'j'), ('d', 'f'), 
                   ('g', 'j'), ('a', 'e'), ('i', 'd')]
        print('练习21.1-1') 
        print(' 无向图G=(V, E)的顶点集合V=')
        print('  {}'.format(g.vertexs))
        print(' 边集合E=')
        print('  {}'.format(g.edges))
        print(' 所有连通子图的顶点集合为')
        print('  {}'.format(g.get_connected_components()))
        print('练习21.1-2 证明：在CONNECTED-COMPONENTS处理了所有的边后')
        print(' 两个顶点在同一个连通子图中，当且仅当它们在同一个集合中')
        print('练习21.1-3 无向图G=(V,E)调用FIND_SET的次数为len(E)*2,',
            '调用UNION的次数为len(V) - k')
        g.print_last_connected_count()
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_2:
    '''
    chpater21.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.2 note

        Example
        ====
        ```python
        Chapter21_2().note()
        ```
        '''
        print('chapter21.2 note as follow')
        print('21.2 不相交集合的链表表示')
        print(' 要实现不相交集合数据结构，一种简单的方法是每一种集合都用一个链表来表示')
        print(' 每个链表中的第一个对象作为它所在集合的代表')
        print(' 链表中的每一个对象都包含一个集合成员，一个指向包含下一个集合成员的对象的指针，以及指向代表的指针')
        print(' 每个链表都含head指针和tail指针，head指向链表的代表，tail指向链表中最后的对象')
        print(' 在这种链表表示中，MAKE-SET操作和FIND-SET操作都比较容易实现，只需要O(1)的时间')
        print(' 执行MAKE-SET(x)操作，创建一个新的链表,其仅有对象为x')
        print(' 对FIND-SET(X)操作,只要返回由x指向代表的指针即可')
        print('合并的一个简单实现')
        print(' 在UNION操作的实现中，最简单的是采用链表集合表示的实现，',
            '这种实现要比MAKE-SET或FIND-SET多不少的时间')
        print(' 执行UNION(x,y),就是将x所在的链表拼接到y所在链表的表尾.利用y所在链表的tail指针',
            '可以迅速地找到应该在何处拼接x所在的链表')
        print(' 一个作用于n个对象上的,包含m个操作的序列，需要Θ(n^2)时间')
        print(' 执行n个MAKE-SET操作所需要的时间为Θ(n)')
        print(' 因为第i个UNION操作更新了i个对象，故n-1个UNION操作所更新的对象总数为Θ(n^2)')
        print(' 总的操作数为2n-1，平均来看，每个操作需要Θ(n)的时间')
        print(' 也就是一个操作的平摊时间为Θ(n)')
        print('一种加权合并启发式策略')
        print(' 在最坏情况下，根据上面给出的UNION过程的实现，每次调用这一过程都需要Θ(n)的时间')
        print(' 如果两个表一样长的话，可以以任意顺序拼接，利用这种简单的加权合并启发式策略')
        print(' 如果两个集合都有Ω(n)个成员的话，一次UNION操作仍然会需要Θ(n)时间')
        print('定理21.1 利用不相交集合的链表表示和加权合并启发式',
            '一个包括m个MAKE-SET,UNION和FIND-SET操作',
            '(其中有n个MAKE-SET操作)的序列所需时间为O(m+nlgn)')
        print('练习21.2-1 已经完成')
        print('练习21.2-2 如下:结果是16个链表集合合并成了一个总的链表')
        _nset.test_list_set()
        print('练习21.2-3 对定理21.1的证明加以改造，使得MAKE-SET和FIND-SET操作有平坦时间界O(1)')
        print(' 对于采用了链表表示和加权合并启发式策略的UNION操作，有界O(lgn)')
        print('练习21.2-4 假定采用的是链表表示和加权合并启发式策略。略')
        print('练习21.2-5 合并的两个表像合并排序那样轮流交叉合并')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_3:
    '''
    chpater21.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.3 note

        Example
        ====
        ```python
        Chapter21_3().note()
        ```
        '''
        print('chapter21.3 note as follow')
        print('21.3 不相交集合森林')
        print('在不相交集合的另一种更快的实现中，用有根树来表示集合，',
            '树中的每个结点都包含集合的一个成员，每棵树表示一个集合')
        print('在不相交集合森林中，每个成员仅指向其父结点。',
            '每棵树的根包含了代表，并且是它自己的父亲结点')
        print('尽管采用了这种表示的直观算法并不比采用链表表示的算法更快')
        print('通过引入两种启发式策略(\"按秩合并\"和\"路径压缩\")',
            '就可以获得目前已知的、渐进意义上最快的不相交集合数据结构了')
        print('不相交集合森林。MAKE-SET创建一棵仅包含一个结点的树')
        print('在执行FIND-SET操作时，要沿着父结点指针一直找下去，直到找到树根为止')
        print('在这一查找路径上访问过的所有结点构成查找路径(find path),UNION操作使得一棵树的根指向另一颗树的根')
        print('改进运行时间的启发式策略')
        print(' 还没有对链表实现做出改进。一个包含n-1次UNION操作的序列可能会构造出一棵为n个结点的线性链的树')
        print(' 通过采用两种启发式策略，可以获得一个几乎与总的操作数m成线性关系的运行时间')
        print(' 1.按秩合并(union by rank),与我们用于链表表示中的加权合并启发式是相似的')
        print('   其思想是使包含较少结点的树的根指向包含较多结点的树的根')
        print('   对每个结点，用秩表示结点高度的一个上界。在按秩合并中，具有较小秩的根的UNION操作中要指向具有较大秩的根')
        print(' 2.路径压缩(path compression),非常简单有效,在FIND-SET操作中,',
            '利用这种启发式策略,来使查找路径上的每个结点都直接指向根结点,路径压缩并不改变结点的秩')
        print('启发式策略对运行时间的影响')
        print(' 如果将按秩合并或路径压缩分开来使用的话，都能改善不相交集合森林操作的运行时间')
        print(' 如果将这两种启发式合起来使用，则改善的幅度更大。')
        print(' 单独来看,按秩合并产生的运行时间为O(mlgn),这个界是紧确的')
        print(' 如果有n个MAKE-SET操作和f个FIND-SET操作,则单独应用路径压缩启发式的话')
        print(' 得到的最坏情况运行时间为Θ(n+f*(1+log2+f/n(n)))')
        print('当同时使用按秩合并和路径压缩时，最坏情况运行时间为O(ma(n)),',
            'a(n)是一个增长及其缓慢的函数')
        print(' 在任意可想象的不相交集合数据结构的应用中，都会有a(n)<=4',
            '在各种实际情况中，可以把这个运行时间看作与m成线性关系')
        print('练习21.3-1 用按秩合并和路径压缩启发式的不相交集合森林重做练习21.2-2')
        _nset.test_forest_set()
        print('练习21.3-2 写出FIND-SET的路径压缩的非递归版本')
        print('练习21.3-3 请给出一个包含m个MAKE-SET，UNION和FIND-SET操作的序列',
              '(其中n个是MAKE-SET操作)，使得采用按秩合并时，这一操作序列的时间代价为Ω(mlgn)')
        print('练习21.3-4 证明：在采用了按秩合并和路径压缩时，',
            '任意一个包含m个MAKE-SET,FIND-SET和LINK操作的序列',
            '(其中所有LINK操作出现于FIND-SET操作之前)需要O(m)的时间')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

class Chapter21_4:
    '''
    chpater21.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter21.4 note

        Example
        ====
        ```python
        Chapter21_4().note()
        ```
        '''
        print('chapter21.4 note as follow')
        print('21.4 带路径压缩的按秩合并的分析')
        # !对作用于n个元素上的m个不相交集合操作，
        # !联合使用按秩合并和路径压缩启发式的运行时间为O(ma(n))
        print('对作用于n个元素上的m个不相交集合操作,',
            '')
        print('联合使用按秩合并和路径压缩启发式的运行时间为O(ma(n))')
        print('秩的性质')
        print('引理21.4 对所有的结点x，有rank[x]<=rank[p[x]],如果x!=p[x]则不等号严格成立')
        print(' rank[x]的初始值位0，并随时间而增长，直到x!=p[x];从此以后rank[x]就不再变化')
        print(' rank[p[x]]的值是时间的单调递增函数')
        print('推论21.5 在从任何一个结点指向根的路径上，结点的秩是严格递增的')
        print('引理21.6 每个结点的秩至多为1')
        print('时间界的证明')
        print('引理21.7 假定有一个m个MAKE-SET,UNION和FIND-SET操作构成的操作序列S\'',
            '将其转换成一个新的操作序列S，它由m个MAKE-SET,LINK和FIND-SET操作所构成',
            '转换方法是将每一个UNION操作转换成两个FIND-SET操作，后跟一个LINK操作',
              '如果操作序列S的运行时间为O(ma(n)),操作序列S\'的运行时间即为O(m\'a(n))')
        print('引理21.10 每个MAKE-SET操作的平摊代价是为O(1)')
        print('引理21.11 每个LINK操作的平摊代价为O(a(n))')
        print('引理21.12 每个FIND-SET操作的平摊代价为O(a(n))')
        print('定理21.13 对于一个由m个MAKE-SET,UNION和FIND-SET操作所组成的操作序列(其中n个为MAKE-SET操作)',
              '当在一个按秩合并和路径压缩的不相交集合森林上执行时，最坏情况下的执行时间O(ma(n))')
        print('练习21.4-1 略')
        print('练习21.4-2 证明：每个结点的秩都至多为[lgn]')
        print('练习21.4-3 对每个结点x，存储rank[x]需要多少位(bit)')
        print('练习21.4-4 对于带按秩合并,但不带路径压缩的不相交集合上的操作，',
            '简要地证明其运行时间为O(mlgn)')
        print('练习21.4-5 各结点的秩在一条指向根的路径上是严格递增的')
        print('练习21.4-6 证明：一个包含m个MAKE-SET,UNION和FIND-SET操作(其中n个为MAKE-SET操作)的序列',
              '当在一个带按秩合并的路径压缩的不相交集合森林上执行时，其最坏情况运行时间为O(ma\'(n))')
        print('思考题21-1 脱机最小值')
        print(' 脱机最小值问题(off-line minimum problem)是对INSERT和EXTRACT-MIN操作所所用的',
            '一个其元素取自域{1,2,..,n}的动态集合T加以维护')
        print(' 已知的是一个包含n个INSERT和m个EXTRACT-MIN调用的序列S,其中{1,2,...,n}',
            '中的每一个关键字恰被插入一次')
        print(' 在下面的脱机最小值问题的例子中，每个INSERT由一个数字表示,每个EXTRACT-MIN由字母E表示')
        print(' 4,8,E,3,E,9,2,6,E,E,E,1,7,E,5')
        print(' 将正确的值填入extracted数组')
        print(' 如何用不相交集合数据结构有效地实现OFF-LINE-MINIMUM')
        print('思考题21-2 深度确定')
        print(' 在深度确定问题中，对一下三个操作所作用的一个有根树的森林F加以维护')
        print(' (1) MAKE-TREE(v):创建一棵包含唯一结点v的树')
        print(' (2) FIND-DEPTH(v):返回结点v在树中的深度')
        print(' (3) GRAFT(r,v)：使结点r(假定为某棵树的根)成为结点v的子结点',
            '(假定结点v在另一棵树中,它本身可能是,也可能不是一个根)')
        print('思考题21-3 Tarjan的脱机最小公共祖先算法')
        print(' 在一棵有根树T中，两个结点u和v的最小公共祖先是指这样的一个结点w,它是u和v的祖先，并且在树T中具有最大深度')
        print(' 在脱机最小公共祖先问题中，给定的是一棵有根树T和一个由T中结点的无序对构成的任意集合P={(u, v)}')
        print(' 希望确定P中每个对的最小公共祖先')
        # python src/chapter21/chapter21note.py
        # python3 src/chapter21/chapter21note.py

chapter21_1 = Chapter21_1()
chapter21_2 = Chapter21_2()
chapter21_3 = Chapter21_3()
chapter21_4 = Chapter21_4()

def printchapter21note():
    '''
    print chapter21 note.
    '''
    print('Run main : single chapter twenty-one!')  
    chapter21_1.note()
    chapter21_2.note()
    chapter21_3.note()
    chapter21_4.note()

# python src/chapter21/chapter21note.py
# python3 src/chapter21/chapter21note.py
if __name__ == '__main__':  
    printchapter21note()
else:
    pass

```

```py

class UndirectedGraph:
    '''
    无向图 `G=(V, E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        无向图 `G=(V, E)`

        Args
        ===
        `vertexs` : 顶点集合 `list` contains element which contains one element denote a point

        `edges` : 边集合 `list` contains element which contains two elements denote one edge of two points repectively

        Example
        ===
        ```python
        import notintersectset as graph
        >>> g = graph.UndirectedGraph(['a', 'b', 'c', 'd'], [('a', 'b')])
        ```
        '''
        self.vertexs = vertexs
        self.edges = edges
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0

    def get_connected_components(self):
        '''
        获取无向图中连通子图的集合
        '''
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0
        set = Set()
        for v in self.vertexs:
            set.make_set(v)
        for e in self.edges:
            u, v = e
            set1 = set.find(u)
            set2 = set.find(v)
            self.__findcount += 2
            if set1 != set2:
                set.union(set1, set2)
                self.__unioncount += 1
        self.__kcount = len(set.sets)
        return set

    def print_last_connected_count(self):
        '''
        获取上一次连接无向图之后调用函数情况
        '''
        print('the k num:{} the find num:{} the union num:{}'. \
            format(self.__kcount, self.__findcount, self.__unioncount))

class Set:
    '''
    不相交集合数据结构
    '''
    def __init__(self):
        '''
        不相交集合数据结构
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        self.sets.append({element})

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        if set1 is None or set2 is None:
            return
        self.sets.remove(set1)
        self.sets.remove(set2)
        self.sets.append(set1 | set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            if element in set:
                return set
        return None
    
    def __str__(self):
        return str(self.sets)

    def printsets(self):
        '''
        打印集合
        '''
        for set in self.sets:
            print(set)

class ListNode:
    def __init__(self, key = None):
        '''
        采用链表表示不相交集合结点
        '''
        self.first = None
        self.next = None
        self.key = key
    
    def __str__(self):
        return str(self.key)

class List:
    def __init__(self):
        '''
        采用链表表示不相交集合
        '''
        self.rep = None
        self.head = None
        self.tail = None
        self.size = 0
    
    def __str__(self):
        return 'List size:{} and rep:{}'.format(self.size, self.rep)

class ListSet(Set):
    '''
    不相交集合的链表表示
    '''
    def __init__(self):
        '''
        不相交集合的链表表示
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        list = List()
        node = ListNode(element)  
        if list.size == 0:           
            list.head = node
            list.tail = node
            list.rep = node
            node.first = node
            list.size = 1
        else:
            list.tail.next = node
            list.tail = node 
            node.first = list.head
            list.size += 1
        self.sets.append(list)

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        self.sets.remove(set1)
        self.sets.remove(set2)
        set1.tail.next = set2.rep
        set1.size += set2.size
        set1.tail = set2.tail

        set2.rep = set1.rep

        node = set2.head
        for i in range(set2.size):
            node.first = set1.rep
            node = node.next

        self.sets.append(set1)

    def unionelement(self, element1, element2):
        '''
        将`element1`代表的集合和`element2`代表的集合合并
        '''
        set1 = self.find(element1)
        set2 = self.find(element2)
        if set1 is None or set2 is None:
            return
        if set1.size < set2.size:
            self.union(set2, set1)
        else:
            self.union(set1, set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            node = set.rep
            while node != set.tail:
                if node.key == element:
                    return set
                node = node.next
            else:
                if set.tail.key == element: 
                    return set
        return None

class RootTreeNode:
    '''
    有根树结点
    '''
    def __init__(self, key = None, parent = None, rank = None):
        '''
        有根树结点

        Args
        ===
        `key` : 关键字值

        `parent` : 结点的父结点

        `rank` : 结点的秩
        '''
        self.key = key
        self.parent = parent
        self.rank = rank
    
    def __str__(self):
        return 'key:{} rank:{}'.format(self.key, self.rank)

class RootTree:
    '''
    有根树
    '''
    def __init__(self, root = None):
        '''
        有根树
        Args
        ===
        `root` : 有根树的根结点
        '''
        self.root = root
    
    def __str__(self):
        return 'roots:' + str(self.root)

class ForestSet(Set):
    '''
    不相交集合森林
    '''
    def __init__(self):
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        treenode = RootTreeNode(element)
        self.make_set_node(treenode)
        
    def make_set_node(self, node : RootTreeNode):
        '''
        用有根树结点`node`建立一个新的集合
        '''
        tree = RootTree()  
        node.parent = node
        node.rank = 0
        tree.root = node
        self.sets.append(tree)

    @classmethod
    def link(self, node1 : RootTreeNode, node2 : RootTreeNode):
        '''
        连接两个有根树的结点`node1`和`node2`
        '''
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1
    
    def union(self, x : RootTreeNode, y : RootTreeNode):
        '''
        将有根树结点`x`代表的集合和有根树结点`y`代表的集合合并
        '''
        self.link(self.findnode(x), self.findnode(y))

    def findnode(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合
        '''
        if x != x.parent:
            x.parent = self.findnode(x.parent)
        return x.parent

    def findnode_nonrecursive(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合(非递归版本)
        '''
        y = x
        while y != y.parent:
            y = y.parent
        while x != x.parent:
            x.parent = y
            x = x.parent

def connected_components(g: UndirectedGraph):
    '''
    求一个无向图中连通子图的个数

    Args
    ===
    `g` : UndirectedGraph 无向图

    '''
    set = Set()
    for v in g.vertexs:
        set.make_set(v)
    for e in g.edges:
        u, v = e
        set1 = set.find(u)
        set2 = set.find(v)
        if set1 != set2:
            set.union(set1, set2)
    return set

def test_graph_connected():
    '''
    测试无向图链接连通子图
    '''
    g = UndirectedGraph()
    g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges.append(('b', 'd'))
    g.edges.append(('e', 'g'))
    g.edges.append(('a', 'c'))
    g.edges.append(('h', 'i'))
    g.edges.append(('a', 'b'))
    g.edges.append(('e', 'f'))
    g.edges.append(('b', 'c'))
    print(g.get_connected_components())
    g.print_last_connected_count()

def test_list_set():
    '''
    不相交集合的链表表示
    '''
    NUM = 16
    set = ListSet()
    for i in range(NUM):
        set.make_set(i)
    for i in range(0, NUM - 1, 2):
        set.unionelement(i, i + 1)
    for i in range(0, NUM - 3, 4):
        set.unionelement(i, i + 2)
    set.printsets()
    set.unionelement(1, 5)
    set.unionelement(11, 13)
    set.unionelement(1, 10)
    set.printsets()
    print(set.find(2))
    print(set.find(9))

def test_forest_set():
    '''
    测试不相交集合森林
    '''
    NUM = 16
    set = ForestSet()
    nodes = []
    for i in range(NUM):
        nodes.append(RootTreeNode(i))
    for i in range(NUM):
        set.make_set_node(nodes[i])
    set.printsets()
    for i in range(0, NUM - 1, 2):
        set.union(nodes[i], nodes[i + 1])
    set.printsets()
    for i in range(0, NUM - 3, 4):
        set.union(nodes[i], nodes[i + 2])
    set.printsets()

if __name__ == '__main__':
    test_graph_connected()
    test_list_set()
    test_forest_set()
else:
    pass    

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
'''

Class Chapter22_1

Class Chapter22_2

Class Chapter22_3

Class Chapter22_4

Class Chapter22_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    import graph as _g
else: 
    from . import graph as _g

class Chapter22_1:
    '''
    chpater22.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.1 note

        Example
        ====
        ```python
        Chapter22_1().note()
        ```
        '''
        print('chapter22.1 note as follow')  
        print('第六部分 图算法')
        print('图是计算机科学中常用的一类数据结构，有关图的算法也是计算机科学中基础性算法')
        print('有许多有趣而定计算问题都是用图来定义的')
        print('第22章介绍图在计算机中的表示，并讨论基于广度优先或深度优先图搜索的算法')
        print(' 给出两种深度优先搜索的应用；根据拓扑结构对有向无回路图进行排序，以及将有向图分解为强连通子图')
        print('第23章介绍如何求图的最小权生成树(minimum-weight spanning tree)')
        print(' 定义：即当图中的每一条边都有一个相关的权值时，',
            '这种树由连接了图中所有顶点的、且权值最小的路径所构成')
        print(' 计算最小生成树的算法是贪心算法的很好的例子')
        print('第24章和第25章考虑当图中的每条边都有一个相关的长度或者\"权重\"时，如何计算顶点之间的最短路径问题')
        print('第24章讨论如何计算从一个给定的源顶点至所有其他顶点的最短路径问题')
        print('第25章考虑每一对顶点之间最短路径的计算问题')
        print('第26章介绍在物流网络(有向图)中，物流的最大流量计算问题')
        print('在描述某一给定图G=(V, E)上的一个图算法的运行时间，通常以图中的顶点个数|V|和边数|E|来度量输入规模')
        print('比如可以讲该算法的运行时间为O(VE)')
        print('用V[G]表示一个图G的顶点集,用E[G]表示其边集')
        print('第22章 图的基本算法')
        print('22.1 图的表示')
        print('要表示一个图G=(V,E),有两种标准的方法，即邻接表和邻接矩阵')
        print('这两种表示法即可以用于有向图，也可以用于无向图')
        print('通常采用邻接表表示法，因为用这种方法表示稀疏图比较紧凑')
        print('但是，当遇到稠密图(|E|接近于|V|^2)或必须很快判别两个给定顶点是否存在连接边，通常采用邻接矩阵表示法')
        print('图G=(V,E)的邻接表表示由一个包含|V|个列表的数组Adj所组成,其中每个列表对应于V中的一个顶点')
        print('对于每一个u∈V，邻接表Adj[u]包含所有满足条件(u,v)∈E的顶点v')
        print('亦即Adj[u]包含图G中所有的顶点u相邻的顶点')
        print('如果G是一个有向图,则所有邻接表的长度之和为|E|,这是因为一条形如',
            '(u,v)的边是通过让v出现在Adj[u]中来表示的')
        print('如果G是一个无向图，则所有邻接表的长度之和为2|E|')
        print('因为如果(u,v)是一条无向边,那么u就会出现在v的邻接表中')
        print('不论是有向图还是无向图，邻接表表示法都有一个很好的特性，即它所需要的存储空间为Θ(V+E)')
        print('邻接表稍作变动，即可用来表示加权图，即每条边都有着相应权值的图')
        print('权值通常由加权函数w给出，例如设G=(V,E)是一个加权函数为w的加权图')
        print('邻接表表示法稍作修改就能支持其他多种图的变体，因而有着很强的适应性')
        print('邻接表表示法也有着潜在不足之处，即如果要确定图中边(u,v)是否存在，',
            '只能在顶点u的邻接表Adj[u]中搜索v,除此之外，没有其他更快的方法')
        print('这个不足可以通过图的邻接矩阵表示法来弥补，但要在(在渐进意义下)以占用更多的存储空间作为代价')
        # !一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的
        print('一个图的邻接矩阵表示需要占用Θ(V^2)的存储空间,与图中的边数多少是无关的')
        print('邻接矩阵是沿主对角线对称的')
        print('正如图的邻接表表示一样，邻接矩阵也可以用来表示加权图')
        print('例如，如果G=(V,E)是一个加权图，其权值函数为w，对于边(u,v)∈E,其权值w(u,v)')
        print('就可以简单地存储在邻接矩阵的第u行第v列的元素中，如果边不存在，则可以在矩阵的相应元素中存储一个None值')
        # !邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的
        print('邻接表表示和邻接矩阵表示在渐进意义下至少是一样有效的')
        print('但由于邻接矩阵简单明了,因而当图较小时,更多地采用邻接矩阵来表示')
        print('另外如果一个图不是加权的，采用邻接军阵的存储形式还有一个优越性:',
            '在存储邻接矩阵的每个元素时，可以只用一个二进制位，而不必用一个字的空间')
        print('练习22.1-1 给定一个有向图的邻接表示，计算该图中每个顶点的出度和入度都为O(V+E)')
        print(' 计算出度和入度的过程相当于将邻接链表的顶点和边遍历一遍')
        print('练习22.1-2 给出包含7个顶点的完全二叉树的邻接表表示，写出其等价的邻接矩阵表示')
        g = _g.Graph()
        g.veterxs = ['1', '2', '3', '4', '5', '6', '7']
        g.edges = [('1', '2'), ('1', '3'), ('2', '4'),
               ('2', '5'), ('3', '6'), ('3', '7')]
        print(g.getmatrix())
        print('练习22.1-3 邻接链表：对于G的每个节点i，遍历；adj,将i添加到adj中遇到的每个结点')
        print(' 时间就是遍历邻接链表的时间O(V+E)')
        print('邻接矩阵：就是求G的转置矩阵，时间为O(V^2)')
        print('练习22.1-4 给定一个多重图G=(V,E)的邻接表表示,给出一个具有O(V+E)时间的算法,',
            '计算“等价”的无向图G`(V,E`)的邻接表，其中E`包含E中所有的边,',
            '且将两个顶点之间的所有多重边用一条边代表，并去掉E中所有的环')
        print('练习22.1-5 算法运行时间都为O(V^3)')
        print('练习22.1-6 当采用邻接矩阵表示时，大多数图算法需要的时间都是Ω(V^2),但也有一些例外')
        print(' 证明：在给定了一个有向图G邻接矩阵后，可以在O(V)时间内，',
            '确定G中是否包含一个通用的汇，即入度|V|-1,出度为0顶点')
        print('练习22.1-7 矩阵乘积对角线上的元素表示与某结点连接的边的个数')
        print(' 若第m行第n列的元素为-1，则第m个结点到第n个结点连通，并且方向从m到n')
        print('练习22.1-8 假设每个数组元素Adj[u]采用的不是链表,而是一个包含了所有满足(u,v)∈E的顶点v的散列表')
        print(' 如果所有的边查找都是等可能的，则确定某条边是否在途中所需的期望时间是多少')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_2:
    '''
    chpater22.2 note and function
    '''
    def buildGraph(self):
        '''
        练习22.2-1

        练习22.2-2
        '''
        g = _g.Graph()
        g.veterxs = [_g.Vertex('1'), _g.Vertex('2'),
                     _g.Vertex('3'), _g.Vertex('4'),
                     _g.Vertex('5'), _g.Vertex('6')]
        g.edges.clear()
        g.edges.append(_g.Edge(_g.Vertex('1'), _g.Vertex('2'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('4'), _g.Vertex('2'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('1'), _g.Vertex('4'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('2'), _g.Vertex('5'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('3'), _g.Vertex('6'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('3'), _g.Vertex('5'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('5'), _g.Vertex('4'), 1, _g.DIRECTION_TO))
        g.edges.append(_g.Edge(_g.Vertex('6'), _g.Vertex('6'), 1, _g.DIRECTION_TO))
        _g.bfs(g, g.veterxs[2])
        _g.print_path(g, g.veterxs[2], g.veterxs[4])
        print('')
        del g
        
        g = _g.Graph()
        g.veterxs.clear()
        g.edges.clear()
        v = ['r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        g.addvertex(v)
        g.addedge('v', 'r')
        g.addedge('r', 's')
        g.addedge('s', 'w')
        g.addedge('w', 'x')
        g.addedge('w', 't')
        g.addedge('x', 't')
        g.addedge('x', 'u')
        g.addedge('x', 'y')
        g.addedge('y', 'u')
        g.addedge('u', 't')
        _g.bfs(g, 'u')
        _g.print_path(g, 'u', 'v')
        print('')
        del g

    def note(self):
        '''
        Summary
        ====
        Print chapter22.2 note

        Example
        ====
        ```python
        Chapter22_2().note()
        ```
        '''
        print('chapter22.2 note as follow')
        print('22.2 广度优先搜索')
        # !广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型
        print('广度优先搜素是最简单的图搜索算法之一,也是很多重要的图算法的原型')
        print('在Prim最小生成树算法和Dijkstra单源最短路径算法，都采用了与广度优先搜索类似的思想')
        # !在给定图G=(V,E)和特定的一个源顶点s的情况下,广度优先搜索系统地探索G中的边
        print('在给定图G=(V,E)和特定的一个源顶点s的情况下,广度优先搜索系统地探索G中的边，',
            '以期发现可从s到达的所有顶点,并计算s到所有这些可达顶点之间的距离(即最少的边数)')
        print('该搜素算法同时还能生成一棵根为s,且包括所有s的可达顶点的广度优先树')
        print('对从s可达的任意顶点v,从优先树中从s到v的路径对应于图G中从s到v的一条最短路径,',
            '即包含最少边数的路径.该算法对有向图和无向图同样适用')
        print('之所以称为广度优先搜索：算法首先会发现和s距离为k的所有顶点,',
            '然后才会发现和s距离为k+1的其他顶点')
        print('为了记录搜索的轨迹，广度优先搜索将每个顶点都着色为白色，灰色或者黑色')
        print('白色表示还未搜索，灰色和黑色表示已经被发现')
        print('与黑色结点相邻的所有顶点都是已经被发现的。')
        print('灰色顶点可能会有一些白色的相邻结点,它们代表了已经发现与未发现顶点之间的边界')
        print('广度优先搜索构造了一棵广度优先树,在开始时只包含一个根顶点，即源顶点s')
        print('在扫描某个已发现顶点u的邻接表的过程中,每当发现一个白色顶点v,',
            '该顶点v以及边(u,v)就被添加到树中')
        print('在广度优先树中,称u是v的先辈或者父母。',
            '由于一个顶点至多只能被发现一次,因此它最多只能有一个父母顶点。',
            '在广度优先树中，祖先和后裔关系的定义和通常一样,是相对于根s来定义的：',
            '如果u处于树中从根s到顶点v的路径中，那么u称为v的祖先，v是u的后裔')
        _g.test_bfs()
        print('只要队列Q中还有灰色顶点(即那些已经被发现，但是还没有)')
        print('广度优先搜索运行时间分析')
        print(' 采用聚集分析技术,由于所有邻接表长度之和为Θ(E).初始化操作的开销为O(V)')
        print(' 过程BFS的总运行时间为O(V+E),由此可见，',
            '广度优先搜索的运行时间是图G的邻接表大小的一个线性函数')
        print('最短路径')
        print(' 对于一个图G=(V,E),(有向图和无向图均可以),广度优先搜索算法可以得到从已知源顶点s∈V到每个可达顶点的距离')
        print(' 定义从顶点s到v之间的最短路径距离d(s,v)为s到v的任何路径中最少的边数')
        print('  如果两个点s到v之间没有同路，则距离为无穷')
        print(' 广度优先搜索计算出来的就是最短路径')
        print('引理22.1 设G=(V,E)是一个有向图或无向图,s∈V为G的任意一个顶点，则对任意边(u,v)∈V,有:d(s,v)<=d(s,u)+1')
        print('证明：如果从顶点s可达顶点u，则从s也可达v.在这种情况下，从s到v的最短路径不可能比s到u的最短路径加上边(u,v)更长',
            '因此不等式成立。如果从s不可达顶点u，则d(s,u)=∞，不等式仍然成立')
        print('引理22.2 设G=(V,E)是一个有向图或无向图,并假设算法BFS(广度优先搜索)从G中某一给定源顶点s∈V开始执行',
            '在执行终止时，对每个顶点v∈V，BFS所计算出来的v.d的值没看组v.d>=d(s,v)')
        print('引理22.3 假设过程BFS在图G=(V,E)上的执行过程中,队列Q包含顶点<v1,v2,..,vr>',
            '其中v1是队列的头，vr是队列的尾巴','则d[vr]<=d[v1]+1,i=1,2,...,r-1')
        print('推论22.4 假设在BFS的执行过程中将顶点vi和vj插入了队列，且vi先于vj入队',
            '那么，当vj入队时，有d[vi]<=d[vj]')
        print('定理22.5 (广度优先搜索的正确性)设G=(V,E)是一个有向图或无向图，',
            '并假设过程BFS从G上某个给定的源顶点s可达的每一个顶点v∈V。在运行终止时，对所有v∈V，',
            'd[v]=d(s,v).此外，对任意从s可达的顶点v≠s,从s到v的最短路径之一是从s到v.pi的最短路径再加上边(v.pi,v)')
        print('广度优先树')
        print('过程BFS在搜索图的同时，也建立了一棵广度优先树。这棵树是由每个顶点中的pi域所表示的')
        print('对于图G=(V,E)及给定的源顶点s，可以更为形式化地定义其前趋子图Gpi=(Vpi,Epi)')
        print('引理22.6 当过程BFS应用于某一有向图或无向图G=(V,E)时，',
              '同时要构造出pi域,使得前趋子图Gpi=(Vpi,Epi)是一棵广度优先树')
        print('PRINT-PATH(G,s,v)过程输出从s到v的最短路径上的所有结点',
              '假设已经运行了BFS来计算最短路径')
        print('练习22.2-1 ')
        print('练习22.2-2 ')
        self.buildGraph()
        print('练习22.2-3 略')
        print('练习22.2-4 在广度优先搜索算法BFS中,赋给顶点u的值d[u]与顶点在邻接表中的次序无关')
        print(' 由BFS计算出来的广度优先树与邻接表中的顺序是有关的')
        print('练习22.2-5 在有向图G=(V,E)中，源顶点s∈V，且树边集合满足对每一顶点v∈V，',
            '从s到v的唯一路径是G中的一条最短路径;然而不论在每个邻接表中各顶点如何排列，',
            '都不能通过在G上运行BFS而产生边集')
        print('练习22.2-6 略')
        print('练习22.2-7 树T=(V,E)的直径定义为max(d(u,v)),亦即，树的直径是树中所有最短路径长度中的最大值',
            '试写出计算树的直径的有效算法，并分析算法的运行时间')
        print(' 用无向图构造树')
        print('练习22.2-8 设G=(V,E)是一个连通的无向图。请给出一个O(V+E)时间的算法，以计算图G中的一条路径',
            '对于E中的每一条边,该路径都恰好在每一个方向上遍历一次')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_3:
    '''
    chpater22.3 note and function
    '''
    def buildGraph(self):
        '''
        练习22.3-1

        练习22.5-2
        '''
        g = _g.Graph()
        g.veterxs.clear()
        g.edges.clear()
        v = ['q', 's', 'v', 'w', 't', 'x', 'y', 'z', 'u', 'r']
        g.addvertex(v)
        g.addedge('q', 's', _g.DIRECTION_TO)
        g.addedge('q', 't', _g.DIRECTION_TO)
        g.addedge('q', 'w', _g.DIRECTION_TO)
        g.addedge('s', 'v', _g.DIRECTION_TO)
        g.addedge('v', 'w', _g.DIRECTION_TO)
        g.addedge('w', 's', _g.DIRECTION_TO)
        g.addedge('t', 'x', _g.DIRECTION_TO)
        g.addedge('t', 'y', _g.DIRECTION_TO)
        g.addedge('x', 'z', _g.DIRECTION_TO)
        g.addedge('y', 'q', _g.DIRECTION_TO)
        g.addedge('z', 'x', _g.DIRECTION_TO)
        g.addedge('u', 'y', _g.DIRECTION_TO)
        g.addedge('r', 'y', _g.DIRECTION_TO)
        g.addedge('r', 'u', _g.DIRECTION_TO)
        _g.dfs(g)
        for e in g.edges:
            u, v = g.getvertexfromedge(e)         
            if u.d < v.d and v.d < v.f and v.f < u.f:
                print("边({},{})是树边或者前向边".format(u.key, v.key))
            elif v.d < u.d and u.d < u.f and u.f < v.f:
                print("边({},{})是反向边".format(u.key, v.key))
            elif v.d < v.f and v.f < u.d and u.d < u.f:
                print("边({},{})是交叉边".format(u.key, v.key))
        print('')
        del g

    def note(self):
        '''
        Summary
        ====
        Print chapter22.3 note

        Example
        ====
        ```python
        Chapter22_3().note()
        ```
        '''
        print('chapter22.3 note as follow')
        print('22.3 深度优先搜索')
        # !深度搜索算法遵循的搜索策略是尽可能\"深\"地搜索一个图
        print('这种搜索算法遵循的搜索策略是尽可能\"深\"地搜索一个图')
        # !在深度优先搜索中，对于最新发现的顶点，如果还有以此为起点而未探测到的边，就沿此边继续探测下去
        print('在深度优先搜索中，对于最新发现的顶点，如果还有以此为起点而未探测到的边，就沿此边继续探测下去',
            '当顶点v的所有边都已经被探寻过后，搜索将回溯到发现顶点v有起始点的那些边')
        print('这一过程一直进行到已发现从源顶点可达的所有顶点时为止')
        print('如果还存在未被发现的顶点，则选择其中一个作为源顶点，并重复以上过程')
        print('整个过程反复进行，直到所有的顶点都被发现时为止')
        print('与广度优先搜索类似，在深度优先搜索中，每当扫描已经发现顶点u的邻接表，',
            '从而发现新顶点v时,就将置v的先辈域pi[v]为u')
        print('与广度优先搜索不同的是，其先辈子图形成一棵树，深度优先搜索产生的先辈子图可以由几棵树组成',
            '因为搜索可能由多个源顶点开始重复进行。因此,在深度优先搜索中,',
            '先辈子图的定义也和广度优先搜索中稍有不同')
        print('深度优先搜索的先辈子图形成了一个由数棵深度优先树所组成的深度优先森林。Epi中边称为树边')
        print('与广度优先搜索类似，在深度优先搜索过程中，也通过对顶点进行着色来表示顶点的状态。')
        print('开始时，每个顶点均为白色，搜索中被发现时即置为灰色，结束时又被置为黑色(既当其邻接表被完全检索之后)')
        print('这一技巧可以保证每一个顶点在搜索结束时，只存在于一棵深度优先树中，因此，这些树是不相交的')
        print('除了创建一个深度优先森林外，深度优先搜索同时为每个顶点加盖时间戳。')
        print('每个顶点v由两个时间戳：当顶点v第一次被发现(并置成灰色)时，记录下第一个时间戳d[v]')
        print('每当结束检查v的邻接表(并置v为黑色)时,记录下第二个时间戳f[v]')
        print('许多图的算法中都用到了时间戳，它们对推算深度优先搜索的进行情况有很大的帮助')
        print('广度优先搜索只能有一个源顶点，而深度优先却可以从多个源顶点开始搜索')
        print('广度搜索通常用于从某个源顶点开始，寻找最短路径(以及相关的先辈子图)')
        print('深度优先搜索通常作为另一个算法中的一个子程序')
        print('深度优先搜索遍历图中所有顶点，发现白色顶点时，调用DFS-VISIT访问该顶点')
        print('调用DFS-VISIT(u)时，顶点u就成为深度优先森林中一棵新树的根')
        print('当DFS返回时，每个顶点u都对应于一个发现时刻u.d和一个完成时刻f[u]')
        print('深度优先搜索的性质')
        print(' 利用深度优先搜索，可以获得有关图结构的有价值的信息。',
            '深度优先搜索最基本的特征也许是它的先辈子图Gpi形成了一个由树所组成的森林')
        print(' 这是因为深度优先树的结构准确反映了递归调用DFS-VISIT的过程')
        print(' 也就是v.pi==u当且仅当在搜索u的邻接表的过程当中,调用了过程DFS-VISIT(v)')
        print(' 此外,在深度优先森林中，顶点v是顶点u的后裔,当且仅当v是u为灰色时发现的')
        print('定理22.7(括号定理) 在对一个(有向或无向)图G=(V,E)的任何深度优先搜索中,',
            '对于图中任意两个顶点u和v,下述三个条件中仅有一个成立')
        print(' 1.区间(d[u],f[u])和区间(d[v],f[v])是完全不相交的,',
            '且在深度优先森林中,u或v都不是对方的后裔')
        print(' 2.区间(d[u],f[u])完全包含于区间(d[v],f[v])中,且在深度优先树中,u是v的后裔')
        print(' 3.区间(d[v],f[v])完全包含于区间(d[u],f[u])中,且在深度优先树中,v是u的后裔')
        print('推论22.8(后裔区间的嵌套) 在一个(有向图或无向)图G中的深度优先森林中,',
            '顶点v是顶点u的后裔,当且仅当d[u]<d[v]<f[v]<f[u]')
        print('定理22.9(白色路径定理) 在一个(有向或无向)图G=(V,E)的深度优先森林中,',
            '顶点v是顶点u的后裔,当且仅当在搜索过程中于时刻d[u]发现u时,可以从顶点u出发,',
            '经过一条完全由白色顶点组成的路径到达v')
        print('边的分类')
        print(' 深度优先搜索另一个令人感兴趣的性质就是可以通过搜索对输入图G=(V,E)的边进行归类')
        print(' 这种归类可以用来收集有关图的很多重要信息')
        print(' 如：一个有向图是无回路的，当且仅当对该图的深度优先搜索没有产生\"反向\"边')
        print('根据在图G上进行深度优先搜索产生的深度优先森林Gpi,可以把图的边分为四种类型')
        print(' (1)树边(tree edge)是深度优先森林Gpi中的边。如果顶点v是在探寻边(u,v)时被首次发现的，',
            '那么(u,v)就是一条树边')
        print(' (2)反向边(black edge)是深度优先树中，连接顶点u到它的某个后裔v的非树边(u,v)')
        print(' (3)正向边(forward edge)是指深度优先树中，连接顶点u到它的某个后裔v的非树边(u,v)')
        print(' (4)交叉边(cross edge)是其他类型的边，存在于同一颗深度优先树中的两个顶点之间,',
            '条件是其中一个顶点不是另一个顶点的祖先。交叉边也可以在不同的深度优先树的顶点之间')
        print('可以对算法DFS做一些修改，使之遇到图中的边时，对其进行分类。算法的核心思想在于每条边(u,v)',
            '当该边被第一次探寻到时，即根据所到达的顶点v的颜色进行分类')
        print(' (1)白色(COLOR_WHITE)表明它是一条树边')
        print(' (2)灰色(COLOR_GRAY)表明它是一条树边')
        print(' (3)黑色(BLACK_COLOR)表明它是一条正向边或交叉边')
        print('在无向图中，由于(u,v)和(v,u)实际上是同一条边,上述的边分类可能会产生歧义')
        print('当出现这种情况时，边被归为分类表中适用的第一种类型，将根据算法的执行过程中，',
            '首先遇到的边是(u,v)还是(v,u)来对其进行分类')
        print('在对一个无向图进行深度优先搜索时，不会出现正向边和交叉边')
        print('定理22.10 在对一个无向图G进行深度优先搜索的过程中,',
            'G的每一条边要么是树边,要么是反向边')
        print('广度优先搜索和深度优先搜索在算法上的一个区别就是遍历邻接表的时候先',
            '遍历元素的后继元素还是把所有的兄弟元素遍历完再遍历后继元素')
        print('练习22.3-1 根据白色路径定理')
        print('练习22.3-2 见如下程序')
        self.buildGraph()
        print('练习22.3-3 略')
        print('练习22.3-4 证明：边(u,v)是一条：')
        print(' (a) 树边或前向边，当且仅当d[u]<d[v]<f[v]<f[u]')
        print(' (b) 反向边，当且仅当d[v]<d[u]<f[u]<f[v]')
        print(' (c) 交叉边,当且仅当d[v]<f[v]<d[u]<f[u]')
        print('练习22.3-5 证明：在一个无向图中，如果是根据在深度优先搜索中,(u,v)和(v,u)',
            '哪一个首先被遇到作为标准来将(u,v)归类为树边或反向边的话，',
            '就等价于根据边分类方案中的各类型的优先级来对它进行分类')
        print('练习22.3-6 重写DFS，利用栈消除递归')
        print('练习22.3-7 在一个有向图G中，如果有一条从u到v的路径，',
            '并且在对G的深度优先搜索中,如果有d[u]<d[v],则在所得到的深度优先森林中,',
            'v是u的一个后裔这一推测不一定正确')
        print('练习22.3-8 略')
        print('练习22.3-9 略')
        print('练习22.3-10 解释在有向图中，对于一个顶点u(即使u在G中既有入边又出边)',
            '是如何会最终落到一棵仅包含u的深度优先树中')
        print('练习22.3-11 证明：对无向图G的深度优先搜索可以用来识别出G的连通分支',
            '且深度优先森林中所包含的树的数量与G中的联通分支的数量一样多')
        print('练习22.3-12 在一个有向图G=(V,E)中，如果u->v蕴含着对所有顶点u,v∈V',
            '至多有一条从u到v的简单路径,则称G是单连通的。给出一个有效的',
            '算法来判定一个有向图是否是单连通的')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_4:
    '''
    chpater22.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.4 note

        Example
        ====
        ```python
        Chapter22_4().note()
        ```
        '''
        print('chapter22.4 note as follow')
        print('22.4 拓扑排序')
        # !对有向图或者无向图G=(V,E)进行拓扑排序后，结果为该图所有顶点的一个线性序列
        print('对有向图或者无向图G=(V,E)进行拓扑排序后，结果为该图所有顶点的一个线性序列')
        print('运用深度优先搜索进行拓扑排序')
        print('拓扑排序不同于在第二部分中讨论的通常意义上的排序')
        print('在很多应用中，有向无回路图用于说明事件发生的先后次序')
        print('引理22.11 一个有向图G是无回路的，当且仅当对G进行深度优先搜索时没有得到反向边')
        print('练习22.4-1 略')
        print('练习22.4-2 因为是无回路有向图，在DFS深度优先搜索的基础上去掉颜色即可')
        _g.test_topological_sort()
        print('练习22.4-3 给定的无向图G=(V,E)中是否包含一个回路.算法运行时间应该为O(V)')
        _g.test_hascircuit()
        print('练习22.4-4 证明：如果一个有向图G包含回路，则拓扑排序能产生一个顶点的排序序列')
        print(' 可以最小化坏边的数目，所谓坏边，即那些与所生成的顶点序列不一致的边')
        print('练习22.4-5 在一个有向无回路图G=(V,E)上，执行拓扑排序的另一种方法是重复地寻找一个人度为0的顶点')
        print(' 将该顶点输出,并将该顶点及其所有的出边从图中删除')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

class Chapter22_5:
    '''
    chpater22.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter22.5 note

        Example
        ====
        ```python
        Chapter22_5().note()
        ```
        '''
        print('chapter22.5 note as follow')
        print('22.5 强连通分支')
        print('深度优先搜索的一种经典应用：把一个有向图分解为各强连通分支',
            '(Strong Connected Component SCC)')
        print('很多有关有向图的算法都是从这种分解步骤开始的')
        print('在分解之后，算法即在每一个强连通分支上独立地运行。')
        print('最后，再根据各个分支之间的关系，将所有的解组合起来')
        print('有向图G=(V,E)的一个强连通分支就是一个最大的顶点集合C属于V',
            '对于C中的每一对顶点u和v,由u->v和v->u，亦即，顶点u和v是互相可达的')
        print('图G=(V,E)的转置G^T=(V,E^T),E^T是由G中的边改变方向后所组成的')
        print('建立G^T的所需时间为O(V+E)')
        print('图G和图G^T具有相同的强连通分支')
        _g.test_scc()
        print('引理22.13 设C和C\'是有向图G=(V,E)两个不同的强连通分支，',
              '设u,v∈C,设u\',v\'∈C\',并假设G中存在着一条通路u->u\',那么G中不可能还同时存在通路v\'->v')
        print('引理22.14 设C和C\'为有向图G=(V,E)中的两个不同的强连通分支',
            '假设有一条边(u,v)∈E,其中u∈C,v∈C\'，则f(C)>f(C\')')
        print('练习22.5-1 当在一个图中加入一条新的边后，其强连通分支的数目会减少(因为没有加入新的顶点)')
        print('练习22.5-2 略')
        print('练习22.5-3 用于强连通分支的算法,即在第二次深度优先搜索中使用原图(而不是其转置图)',
            '并按完成时间递增的顺序来扫描各个顶点.说法正确')
        print('练习22.5-4 G^T的分支图的转置与G的分支图是一样的')
        print('练习22.5-5 给出一个O(V+E)时间的算法,以计算一个有向图G=(V,E)的分支图',
            '注意在算法产生的分支图中，两个顶点之间至多只能有一条边')
        print('练习22.5-6 给定一个有向图G=(V,E),解释如何生成另一个图G\'=(V\',E\')')
        print('练习22.5-7 广度优先搜索每个顶点即可知道有向图两两顶点之间是否存在路径')
        print('思考题22-1 广度优先搜索BFS和深度优先搜索DFS一样都可以对图的边进行分类')
        print(' 深度优先森林把图的边分为树边、正向边、反向边和交叉边四种类型')
        print(' a)证明在对无向图的广度优先搜索中，存在下列性质:')
        print('  1) 不存在正向边和反向边')
        print('  2) 对于每条树边(u,v),有d[v]=d[u]+1')
        print('  3) 对于每条交叉边(u,v),有d[v]=d[u]或者d[v]=d[u]+1')
        print(' b)证明在对有向图的广度优先搜索中，下列性质成立')
        print('  1) 不存在正向边')
        print('  2) 对于每一条树边(u,v),有d[v]=d[u]+1')
        print('  3) 对于每一条交叉边(u,v),有d[v]<=d[u]+1')
        print('  4) 对于每一条反向边(u,v),有0<=d[v]<=d[u]')
        print('思考题22-2 挂接点、桥以及双连通分支')
        print(' 设G=(V,E)是一个无向连通图,如果去掉G的某个顶点后G就不再是连通图了,这样的顶点称为挂接点',
            '如果去掉某一边后,G就不再成为连通图了,这样的边称为桥(bridge)')
        print(' G的双连通分支是满足以下条件的一个最大边集，即该集合中的任意两条边都位于同一个公共简单回路上',
            '可以用深度优先搜索来确定挂接点，桥以及双连通分支')
        print('思考题22-3 欧拉回路')
        print(' 有向强连通图G=(V,E)的欧拉回路是指通过G中每条边仅一次(但可以访问某个顶点多次)的一个回路')
        print(' a) 证明：图G具有欧拉回路，当且仅当每一个顶点v∈V的入度和出度都相等')
        print(' b) 给出一个O(E)时间的算法,它能够在图G中存在着欧拉回路的情况下,找出这一回路')
        print('思考题22-4 可达性')
        print(' 设G=(V,E)是一个有向图,图中每个顶点u∈V都标记有唯一的整数L(u),该整数取自集合={1,2,...,|V|}')
        print(' 对每个顶点v∈V,设R(u)={v∈V：u->v}为从u可达的顶点集合')
        print(' 定义min(u)为R(u)中标记值最小的顶点。亦即min(u)是这样的一个顶点v，',
            '使得L(v)=min{L(w):w∈R(u)}.请给出一个O(V+E)时间的算法,对所有的顶点u∈V,该算法可以计算出min(u)')
        print('对于稀疏矩阵，对于稀疏矩阵，与邻接矩阵表示相比，采用邻接表示法要更好一些')
        print('20世纪50年代后期以来,深度优先搜索得到了广泛的应用,尤其是用在人工智能程序中')
        # python src/chapter22/chapter22note.py
        # python3 src/chapter22/chapter22note.py

chapter22_1 = Chapter22_1()
chapter22_2 = Chapter22_2()
chapter22_3 = Chapter22_3()
chapter22_4 = Chapter22_4()
chapter22_5 = Chapter22_5()

def printchapter22note():
    '''
    print chapter22 note.
    '''
    print('Run main : single chapter twenty-two!')  
    chapter22_1.note()
    chapter22_2.note()
    chapter22_3.note()
    chapter22_4.note()
    chapter22_5.note()

# python src/chapter22/chapter22note.py
# python3 src/chapter22/chapter22note.py
if __name__ == '__main__':  
    printchapter22note()
else:
    pass

```

```py

import math as _math
from copy import deepcopy as _deepcopy

import numpy as _np

COLOR_WHITE = 0
COLOR_GRAY = 1
COLOR_BLACK = 2

DIRECTION_NONE = ' '
DIRECTION_TO = '→'
DIRECTION_FROM = '←'
DIRECTION_BOTH = '←→'

class Vertex:
    '''
    图的顶点
    '''
    def __init__(self, key = None):
        '''
        图的顶点

        Args
        ===
        `key` : 顶点关键字

        '''
        self.key = key
        self.color = COLOR_WHITE
        self.d = _math.inf
        self.pi = None
        self.f = _math.inf

    def resetpara(self):
        '''
        复位所有属性
        '''
        self.color = COLOR_WHITE
        self.d = _math.inf
        self.pi = None
        self.f = _math.inf

    def __str__(self):
        return '[key:{} color:{} d:{} f:{} pi:{}]'.format(self.key, \
            self.color, self.d, self.f, self.pi)

class Edge:
    '''
    图的边，包含两个顶点
    '''
    def __init__(self, vertex1 : Vertex = None, \
            vertex2 : Vertex = None, \
            distance = 1, \
            dir = DIRECTION_NONE,
            ):
        '''
        图的边，包含两个顶点

        Args
        ===
        `vertex1` : 边的一个顶点
        
        `vertex2` : 边的另一个顶点

        `dir` : 边的方向   
            DIRECTION_NONE : 没有方向
            DIRECTION_TO : `vertex1` → `vertex2`
            DIRECTION_FROM : `vertex1` ← `vertex2`
            DIRECTION_BOTH : `vertex1` ←→ `vertex2`

        '''
        self.dir = dir
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.distance = distance
        self.weight = 1

    def __str__(self):
        return str((self.vertex1, self.vertex2, self.dir))

    def __lt__(self, other):
        if type(other) is Graph:
            return self.weight < other.weight 
        else:
            return self.weight < other

    def __gt__(self, other):
        if type(other) is Graph:
            return self.weight > other.weight 
        else:
            return self.weight > other

    def __le__(self, other):
        if type(other) is Graph:
            return self.weight <= other.weight
        else:
            return self.weight <= other

    def __ge__(self, other):
        if type(other) is Graph:
            return self.weight >= other.weight
        else:
            return self.weight >= other

    def __eq__(self, other):
        if type(other) is Graph:
            return self.weight == other.weight
        else:
            return self.weight == other

    def __ne__(self, other):
        if type(other) is Graph:
            return self.weight != other.weight
        else:
            return self.weight != other

class Graph:
    '''
    图`G=(V,E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        图`G=(V,E)`

        Args
        ===
        `vertexs` : 图的顶点

        `edges` : 图的边

        '''
        self.veterxs = vertexs
        self.edges = edges
        self.adj = []
        self.matrix = []
   
    def hasdirection(self):
        '''
        图`g`是否是有向图
        '''
        dir = False
        for i in range(len(self.edges)):
            dir = dir or self.edges[i].dir != DIRECTION_NONE
        return dir

    def veterxs_atkey(self, key):
        '''
        从顶点序列`vertexs`中返回键值为`key`的顶点
        '''
        if type(key) is Vertex:
            return key
        for i in range(len(self.veterxs)):
            if self.veterxs[i].key == key:
                return self.veterxs[i]

    def getvertexadj(self, v : Vertex):
        '''
        获取图中顶点`v`的邻接顶点序列
        '''
        v = self.veterxs_atkey(v)
        if v is None:
            return None
        self.adj = self.getadj()
        self.matrix = self.getmatrix()
        uindex = 0
        for i in range(len(self.veterxs)):
            if self.veterxs[i].key == v.key:
                uindex = i
                break
        return self.adj[uindex]

    def reset_vertex_para(self):
        '''
        复位所有顶点的参数
        '''
        for i in range(len(self.veterxs)):
            self.veterxs[i].resetpara()

    def addvertex(self, v):
        '''
        向图中添加结点`v`
        '''
        if type(v) is list:
            for node in v:
                if type(node) is not Vertex:
                    key = node
                    node = Vertex(key)
                    self.veterxs.append(node)
            return
        if type(v) is not Vertex:
            key = v
            v = Vertex(key)
        self.veterxs.append(v)

    def addedge(self, v1, v2, dir = DIRECTION_NONE):
        '''
        向图中添加边`edge`

        Args
        ===
        `v1` : 边的一个顶点

        `v2` : 边的另一个顶点

        `dir` : 边的方向
            DIRECTION_NONE : 没有方向
            DIRECTION_TO : `vertex1` → `vertex2`
            DIRECTION_FROM : `vertex1` ← `vertex2`
            DIRECTION_BOTH : `vertex1` ←→ `vertex2`
        '''
        egde = Edge(Vertex(v1), Vertex(v2), 1, dir)
        self.edges.append(egde)

    def getvertexfromedge(self, edge : Edge):
        '''
        获取边的两个顶点的引用

        Args
        ===
        `edge` : 边 
        '''
        n = len(self.veterxs)
        if type(edge) is Edge:
            u, v, dir = edge.vertex1, edge.vertex2, edge.dir
            for k in range(n):
                if self.veterxs[k].key == u.key:
                    uindex = k
                if self.veterxs[k].key == v.key:
                    vindex = k
            return (self.veterxs[uindex], self.veterxs[vindex])
        elif len(edge) == 2:
            u, v = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)
        else:
            u, v, dir = edge
            uindex = self.veterxs.index(u)
            vindex = self.veterxs.index(v)
        return (u, v)

    def getadj(self):
        '''
        获取邻接表
        '''
        adj = []
        n = len(self.veterxs)
        if n == 0:
            return []
        for i in range(n):    
            sub = []     
            for edge in self.edges:
                dir = ' '
                if type(edge) is Edge:
                    u, v, dir = edge.vertex1, edge.vertex2, edge.dir
                    for k in range(n):
                        if self.veterxs[k].key == u.key:
                            uindex = k
                        if self.veterxs[k].key == v.key:
                            vindex = k
                elif len(edge) == 2:
                    u, v = edge
                    uindex = self.veterxs.index(u)
                    vindex = self.veterxs.index(v)
                else:
                    u, v, dir = edge
                    uindex = self.veterxs.index(u)
                    vindex = self.veterxs.index(v)
                if dir == DIRECTION_TO and uindex == i:
                    val = self.veterxs[vindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_FROM and vindex == i:
                    val = self.veterxs[uindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_NONE and uindex == i:
                    val = self.veterxs[vindex]
                    if sub.count(val) == 0:
                        sub.append(val)
                elif dir == DIRECTION_NONE and vindex == i:
                    val = self.veterxs[uindex]
                    if sub.count(val) == 0:
                        sub.append(val)               
            adj.append(sub)
        self.adj = adj
        return adj

    def getmatrix(self):
        '''
        获取邻接矩阵,并且其是一个对称矩阵
        '''
        n = len(self.veterxs)
        if n == 0:
            return []
        mat = _np.zeros((n, n))
        for edge in self.edges:
            dir = ' '
            if type(edge) is Edge:
                u, v, dir = edge.vertex1, edge.vertex2, edge.dir 
                for k in range(n):
                    if self.veterxs[k].key == u.key:
                        uindex = k
                    if self.veterxs[k].key == v.key:
                        vindex = k
            elif len(edge) == 2:
                u, v = edge
                uindex = self.veterxs.index(u)
                vindex = self.veterxs.index(v)
            else:
                u, v, dir = edge
                uindex = self.veterxs.index(u)
                vindex = self.veterxs.index(v)                         
            if dir == DIRECTION_TO:
                mat[uindex, vindex] = 1
            elif dir == DIRECTION_FROM:
                mat[vindex, uindex] = 1
            else:
                mat[uindex, vindex] = 1
                mat[vindex, uindex] = 1
        self.matrix = mat
        return mat

    def gettranspose(self):
        '''
        获取图`g`的转置
        '''
        g_rev = _deepcopy(self)
        for i in range(len(g_rev.edges)):
            lastdir = g_rev.edges[i].dir
            g_rev.edges[i].dir = self.__get_rev_dir(lastdir)
        return g_rev

    def __get_rev_dir(self, dir):
        if dir == DIRECTION_FROM:
            dir = DIRECTION_TO
        elif dir == DIRECTION_TO:
            dir = DIRECTION_FROM
        else:
            dir = DIRECTION_NONE
        return dir

    def buildrevedges(self):
        '''
        构造反向的有向图边
        '''
        newedges = []
        n = len(self.edges)
        for i in range(n):
            edge = self.edges[i]
            if type(edge) is Edge:
                v1, v2, dir = edge.vertex1, edge.vertex2, edge.dir
            else:
                v1, v2, dir = edge
            edge_rev = v2, v1, self.__get_rev_dir(dir)
            newedges.append(edge_rev)
        return newedges

    def __buildBMatrix(self, B, v, i, j, v1, v2, dir):
        if v1 != v and v2 != v:
            B[i][j] = 0
        elif v1 == v and dir == DIRECTION_TO:
            B[i][j] = -1
        elif v2 == v and dir == DIRECTION_FROM:
            B[i][j] = -1
        elif v1 == v and dir == DIRECTION_FROM:
            B[i][j] = 1
        elif v2 == v and dir == DIRECTION_TO:
            B[i][j] = 1

    def buildBMatrix(self):
        '''
        构造关联矩阵
        '''
        m = len(self.veterxs)
        n = len(self.edges)
        B = _np.zeros((m, n))
        revedges = self.buildrevedges()
        for i in range(m):
            v = self.veterxs[i]
            for j in range(n):
                edge = self.edges[j]
                if type(edge) is Edge:
                    v1, v2, dir = edge.vertex1, edge.vertex2, edge.dir
                else:
                    v1, v2, dir = edge
                self.__buildBMatrix(B, v, i, j, v1, v2, dir)
            for j in range(n):
                v1, v2, dir = revedges[j]
                self.__buildBMatrix(B, v, i, j, v1, v2, dir)
        return _np.matrix(B)
    
    def contains_uni_link(self):
        '''
        确定有向图`G=(V,E)`是否包含一个通用的汇(入度为|V|-1,出度为0的点)
        '''
        n = len(self.veterxs)
        self.getmatrix()
        m = self.matrix
        for i in range(n):
            if sum(m[i]) == n - 1:
                return True
        return False

    @property
    def has_cycle(self):
        '''
        判断图是否有环路
        '''
        return hascircuit(self)
    
    @property
    def vertex_num(self):
        '''
        返回图中顶点数量
        '''
        return len(self.veterxs)

    @property
    def edge_num(self):
        '''
        返回图中边的数量
        '''
        return len(self.edges)

def bfs(g : Graph, s : Vertex):
    '''
    广度优先搜索(breadth-first search) 时间复杂度`O(V+E)`

    Args
    ===
    `g` : type:`Graph`,图`G(V,E)`(无向图或者有向图均可)

    `s` : type:`Vertex`，搜索的起点顶点

    Return
    ===
    None

    Example
    ===
    ```python
    from graph import *
    g = Graph()
    v = [Vertex('a'), Vertex('b'), Vertex('c'), Vertex('d'), Vertex('e')]
    g.veterxs = v
    g.edges.append(Edge(v[0], v[1]))
    g.edges.append(Edge(v[0], v[2]))
    g.edges.append(Edge(v[1], v[3]))
    g.edges.append(Edge(v[2], v[1]))
    g.edges.append(Edge(v[3], v[0]))
    g.edges.append(Edge(v[4], v[3]))
    print('邻接表为')
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
    for i in range(len(v)):
        bfs(g, v[i])
        print('{}到各点的距离为'.format(v[i]))
        for u in g.veterxs:
            print(u.d, end=' ')
        print(' ')
    ```
    '''
    g.reset_vertex_para()
    adj = g.getadj()
    # g.changeVEToClass()
    if type(s) is not Vertex:
        key = s
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                s = g.veterxs[i]
    n = len(g.veterxs)
    for i in range(n):
        u = g.veterxs[i]
        if type(u) is Vertex:
            u.color = COLOR_WHITE
            u.d = _math.inf
            u.pi = None
        else:
            return
    s.color = COLOR_GRAY
    s.d = 0
    s.pi = None
    q = []
    q.append(s)
    while len(q) != 0:
        u = q.pop(0)
        uindex = 0
        for i in range(n):
            if g.veterxs[i].key == u.key:
                uindex = i
        for i in range(len(adj[uindex])):
            v = adj[uindex][i]
            if v.color == COLOR_WHITE:
                v.color = COLOR_GRAY
                v.d = u.d + 1
                v.pi = u
                q.append(v)
        u.color = COLOR_BLACK

class _DFS:
    def __init__(self):
        self.__adj = []
        self.__sort_list = []
        self.__time = 0
        self.__n = 0
        self.__count = 0
        self.__scc_count = 0
        self.__scc_list = []

    def search_path(self, g: Graph, u: Vertex, k : Vertex):
        '''
        寻找图`g`中顶点`u`到`k`的路径
        '''
        uindex = 0
        for i in range(self.__n):
            if g.veterxs[i].key == u.key:
                uindex = i
                break   
        for i in range(len(self.__adj[uindex])):
            v = self.__adj[uindex][i]
            if v.key == k.key:
                self.__count += 1
            else:
                self.search_path(g, v, k)
        
    def dfs_visit_non_recursive(self, g: Graph, u : Vertex):
        '''
        深度优先搜索从某个顶点开始(非递归)
        '''
        stack = []
        stack.append(u)
        self.__time += 1
        u.d = self.__time
        while len(stack) > 0:
            w = stack.pop(0)
            w.color = COLOR_GRAY            
            uindex = 0
            for i in range(self.__n):
                if g.veterxs[i].key == w.key:
                    uindex = i
                    break     
            for i in range(len(self.__adj[uindex])):
                v = self.__adj[uindex][i]
                if v.color == COLOR_WHITE:
                    v.pi = w
                    stack.append(v)
                    self.__time += 1
                    v.d = self.__time
            w.color = COLOR_BLACK
            self.__time += 1
            w.f = self.__time
        u.color = COLOR_BLACK
        self.__time += 1
        u.f = self.__time

    def dfs_visit(self, g: Graph, u: Vertex):
        '''
        深度优先搜索从某个顶点开始
        '''
        u.color = COLOR_GRAY
        self.__time += 1
        u.d = self.__time
        uindex = 0
        for i in range(self.__n):
            if g.veterxs[i].key == u.key:
                uindex = i
                break
        for i in range(len(self.__adj[uindex])):
            v = self.__adj[uindex][i]
            if v.color == COLOR_WHITE:
                v.pi = u
                self.dfs_visit(g, v)
        u.color = COLOR_BLACK
        self.__time += 1
        u.f = self.__time
        self.__sort_list.append(u)

    def dfs(self, g: Graph):
        '''
        深度优先搜索算法(depth-first search) 时间复杂度`Θ(V+E)`

        Args
        ===
        `g` : type:`Graph`,图`G(V,E)`(无向图或者有向图均可)

        Return
        ===
        None

        Example
        ===
        ```python
        ```
        '''
        self.__adj = g.getadj()
        self.__n = len(g.veterxs)
        self.__time = 0
        self.__sort_list.clear()
        for i in range(self.__n):
            u = g.veterxs[i]
            u.color = COLOR_WHITE
            u.pi = None
        for i in range(self.__n):
            u = g.veterxs[i]
            if u.color == COLOR_WHITE:
                self.dfs_visit(g, u)
    
    def topological_sort(self, g: Graph):
        '''
        拓扑排序 时间复杂度`Θ(V+E)`

        Args
        ===
        `g` : type:`Graph`,图`G(V,E)`(无向图)

        Return
        ===
        `list` : list 排序好的顶点序列

        Example
        ===
        ```python
        import graph as _g
        g = _g.Graph()
        g.vertexs = ...
        g.edges = ...
        topological_sort(g)
        ```
        '''
        self.__sort_list.clear()
        self.dfs(g)
        sort_list = self.__sort_list
        return sort_list

    def getpathnum_betweentwovertex(self, g: Graph, v1: Vertex, v2: Vertex):
        '''
        获取有向无回路图`g`中两个顶点`v1`和`v2`之间的路径数目 时间复杂度`Θ(V+E)`
        '''
        if g.hasdirection() == False:
            print('para g 是无向图，不返回路径')
            return 0
        count = 0
        g.reset_vertex_para()
        adj = g.getadj()
        n = len(g.veterxs)
        if type(v1) is not Vertex:
            key = v1
            for i in range(len(g.veterxs)):
                if g.veterxs[i].key == key:
                    v1 = g.veterxs[i]
        if type(v2) is not Vertex:
            key = v2
            for i in range(len(g.veterxs)):
                if g.veterxs[i].key == key:
                    v2 = g.veterxs[i]
        self.__count = 0
        self.__adj = g.getadj()
        self.__n = len(g.veterxs)
        self.__time = 0
        self.search_path(g, v1, v2)
        return self.__count

    def scc(self, g : Graph):
        '''
        获取图`g`的强连通分支 时间复杂度`Θ(V+E)`
        '''
        self.__scc_count = 0
        self.__scc_list.clear()
        n = len(g.veterxs)
        g.reset_vertex_para()
        list = self.topological_sort(g)
        self.__scc_count += 1
        g_rev = g.gettranspose()
        g_rev.reset_vertex_para()
        self.dfs(g_rev)
        return self.__scc_list, self.__scc_count

__dfs_instance = _DFS()
# 深度优先搜索
dfs = __dfs_instance.dfs
# 拓扑排序
topological_sort = __dfs_instance.topological_sort
# 获得有向无环图的两个顶点间的路径个数
getpathnum_betweentwovertex = __dfs_instance.getpathnum_betweentwovertex
# 强连通分支图
scc = __dfs_instance.scc

def hascircuit_vertex(g: Graph, v : Vertex):
    '''
    判断一个无向图`g`中顶点`v`是否包含连接自己的回路 
    '''
    stack = []
    stack.append(v)
    while len(stack) > 0:      
        stack_v = stack.pop(0) 
        v_adj = g.getvertexadj(stack_v)
        stack_v.color = COLOR_GRAY
        for i in range(len(v_adj)):
            v_next = v_adj[i]
            if v_next.color == COLOR_WHITE:
                v_next.pi = stack_v
                stack.append(v_next) 
            if v_next.key == v.key and stack_v.pi is not None and stack_v.pi.key != v.key:
                return True                
        stack_v.color = COLOR_BLACK
    return False

def hascircuit(g : Graph):
    '''
    判断一个无向图`g`中是否包含回路 时间复杂度`O(V)`
    '''
    n = len(g.veterxs)
    result = False
    for i in range(n):
        v = g.veterxs[i]
        g.reset_vertex_para()
        result = result or hascircuit_vertex(g, v)
        if result == True:
            return True
    return result

def print_path(g : Graph, s : Vertex, v : Vertex):
    '''
    输出图`g`中顶点`s`到`v`的最短路径上的所有顶点

    '''
    g.reset_vertex_para()
    bfs(g, s)
    if type(s) is not Vertex:
        key = s
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                s = g.veterxs[i]
    if type(v) is not Vertex:
        key = v
        for i in range(len(g.veterxs)):
            if g.veterxs[i].key == key:
                v = g.veterxs[i]
    if v == s:
        print('{}→'.format(s.key), end='')
    elif v.pi == None:
        print('no path from {} to {} exists'.format(s.key, v.key))
    else:
        print_path(g, s, v.pi)
        print('{}→'.format(v.key), end='')

def undirected_graph_test():
    '''
    测试无向图
    '''
    g = Graph()
    g.veterxs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges = [('a', 'b'), ('a', 'c'), ('b', 'd'),
               ('b', 'e'), ('c', 'f'), ('c', 'g')]
    print('邻接表为')
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())

def directed_graph_test():
    '''
    测试有向图
    '''
    g = Graph()
    g.veterxs = ['1', '2', '3', '4', '5', '6']
    g.edges = [('1', '2', '→'), ('4', '2', '→'), 
               ('1', '4', '→'), ('2', '5', '→'),
               ('3', '6', '→'), ('3', '5', '→'),
               ('5', '4', '→'), ('6', '6', '→')]
    print('邻接表为')
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
    B = g.buildBMatrix()
    print('关联矩阵为')
    print(B)
    print(B * B.T)
    print('是否包含通用的汇', g.contains_uni_link())

def test_bfs():
    '''
    测试广度优先搜索方法
    '''
    g = Graph()
    v = [Vertex('a'), Vertex('b'), Vertex('c'), Vertex('d'), Vertex('e')]
    g.veterxs = v
    g.edges.clear()
    g.edges.append(Edge(v[0], v[1]))
    g.edges.append(Edge(v[0], v[2]))
    g.edges.append(Edge(v[1], v[3]))
    g.edges.append(Edge(v[2], v[1]))
    g.edges.append(Edge(v[3], v[0]))
    g.edges.append(Edge(v[4], v[3]))
    print('邻接表为')
    print(g.getadj())
    print('邻接矩阵为')
    print(g.getmatrix())
    for i in range(len(v)):
        bfs(g, v[i])
        print('{}到各点的距离为'.format(v[i]))
        for u in g.veterxs:
            print(u.d, end=' ')
        print(' ')
    bfs(g, v[0])
    print_path(g, v[0], v[4])
    print('')
    del g

    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4], 1, DIRECTION_TO))
    print('邻接表为')
    print(gwithdir.getadj())
    print('邻接矩阵为')
    print(gwithdir.getmatrix())
    for i in range(len(vwithdir)):
        bfs(gwithdir, vwithdir[i])
        print('{}到各点的距离为'.format(vwithdir[i]))
        for u in gwithdir.veterxs:
            print(u.d, end=' ')
        print('')
    bfs(gwithdir, vwithdir[0])
    print_path(gwithdir, vwithdir[0], vwithdir[4])
    print('')
    del gwithdir

def test_dfs():
    '''
    测试深度优先搜索方法
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[0], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[4], vwithdir[3], 1, DIRECTION_FROM))
    print('邻接表为')
    print(gwithdir.getadj())
    print('邻接矩阵为')
    print(gwithdir.getmatrix())
    dfs(gwithdir)
    print('')
    del gwithdir

def _print_inner_conllection(collection : list, end='\n'):
    '''
    打印列表内部内容
    '''
    print('[',end=end)
    for i in range(len(collection)):
        print(str(collection[i]), end=end)
    print(']')

def test_topological_sort():
    '''
    测试拓扑排序
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2], 1, DIRECTION_TO))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4], 1, DIRECTION_TO))
    print('邻接表为')
    print(gwithdir.getadj())
    print('邻接矩阵为')
    print(gwithdir.getmatrix())
    sort_list = topological_sort(gwithdir)
    _print_inner_conllection(sort_list)
    print('')
    print('a到e的路径个数为：', getpathnum_betweentwovertex(gwithdir, 'a', 'e'))

def test_hascircuit():
    '''
    测试是否包含环路函数
    '''
    gwithdir = Graph()
    vwithdir = [Vertex('a'), Vertex('b'), Vertex('c'),
                Vertex('d'), Vertex('e')]
    gwithdir.veterxs = vwithdir
    gwithdir.edges.clear()
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[1]))
    gwithdir.edges.append(Edge(vwithdir[1], vwithdir[2]))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[3]))
    gwithdir.edges.append(Edge(vwithdir[3], vwithdir[4]))
    print('是否包含环路：', hascircuit(gwithdir))
    gwithdir.edges.append(Edge(vwithdir[0], vwithdir[2]))
    gwithdir.edges.append(Edge(vwithdir[2], vwithdir[4]))
    print('是否包含环路：', hascircuit(gwithdir))

def test_scc():
    '''
    测量强连通分支算法
    '''
    g = Graph()
    g.veterxs.clear()
    g.edges.clear()
    v = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    g.addvertex(v)
    g.addedge('a', 'b', DIRECTION_TO)
    g.addedge('b', 'c', DIRECTION_TO)
    g.addedge('c', 'd', DIRECTION_TO)
    g.addedge('d', 'c', DIRECTION_TO)
    g.addedge('e', 'a', DIRECTION_TO)
    g.addedge('b', 'e', DIRECTION_TO)
    g.addedge('b', 'f', DIRECTION_TO)
    g.addedge('c', 'g', DIRECTION_TO)
    g.addedge('d', 'h', DIRECTION_TO)
    g.addedge('h', 'd', DIRECTION_TO)
    g.addedge('e', 'f', DIRECTION_TO)
    g.addedge('f', 'g', DIRECTION_TO)
    g.addedge('g', 'f', DIRECTION_TO)
    g.addedge('h', 'g', DIRECTION_TO)
    scc(g)

def test():
    undirected_graph_test()
    directed_graph_test()
    test_bfs()
    test_dfs()
    test_topological_sort()
    test_hascircuit()
    test_scc()

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
'''

Class Chapter24_1

Class Chapter24_2


'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    import mst as _mst
else: 
    from . import mst as _mst

class Chapter23_1:
    '''
    chpater23.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.1 note

        Example
        ====
        ```python
        Chapter23_1().note()
        ```
        '''
        print('chapter23.1 note as follow')  
        print('设计电子线路时，如果要使n个引脚互相连通,可以使用n-1条连接线',
            '每条连接线连接两个引脚。在各种链接方案中，通常希望找出连接线最少的接法')
        print('可以把这一接线问题模型化为一个无向连通图G=(V,E)',
            '其中V是引脚集合，E是每对引脚之间可能互联的集合')
        print('对图中每一条边(u,v)∈E,都有一个权值w(u,v)表示连接u和v的代价(需要的接线数目)')
        print('希望找出一个无回路的子集T∈E,它连接了所有的顶点，且其权值之和w(T)=∑w(u,v)最小')
        print('因为T无回路且连接了所有的顶点,所以它必然是一棵树，称为生成树')
        print('因为由最小生成树可以\"生成\"图G')
        print('把确定树T的问题称为最小生成树问题')
        print('最小生成树问题的两种算法：Kruskal算法和Prim算法')
        print('这两种算法中都使用普通的二叉堆，都很容易达到O(ElgV)的运行时间')
        print('通过采用斐波那契堆，Prim算法的运行时间可以减小到O(E+VlgV),',
            '如果|V|远小于|E|的话,这将是对该算法的较大改进')
        print('这两个算法都是贪心算法，在算法的每一步中，都必须在几种可能性中选择一种')
        print('贪心策略的思想是选择当时最佳的可能性，一般来说，这种策略不一定能保证找到全局最优解')
        print('然而，最小生成树问题来说,却可以证明某些贪心策略的确可以获得具有最小权值的生成树')
        print('23.1 最小生成树的形成')
        print('假设已知一个无向连通图G=(V,E),其权值函数为w')
        print('目的是找到图G的一棵最小生成树')
        print('通用最小生成树算法')
        print('在每一个步骤中都形成最小生成树的一条边,算法维护一个边的集合A,保持以下的循环不变式:')
        print(' 在每一次循环迭代之前，A是某个最小生成树的一个子集')
        print(' 在算法的每一步中，确定一条边(u,v)，使得将它加入集合A后，仍然不违反之歌循环不变式;',
            '亦即，A∪{(u,v)}仍然是某一个最小生成树的子集')
        print(' 称这样的边为A的安全边(safe edge),因为可以安全地把它添加到A中,而不会破坏上述的循环不变式')
        print('在算法的执行过程中，集合A始终是无回路的，否则包含A的最小生成树将包含一个环')
        print('无向图G=(V, E)的一个割(S, V-S)是对V的一个划分.当一条边(u,v)∈E的一个端点属于S，而另一个端点属于V-S',
            '则称边(u,v)通过割(S,V-S).如果一个边的集合A中没有边通过某一割','则说该割不妨害边集A')
        print('如果某条边的权值是通过一个割的所有边中最小的,则称该边为通过这个的割的一条轻边(light edge)')
        print('GENERIC-MST')
        print('  A = []')
        print('  while A does not form a spanning tree')
        print('    do find an edge (u,v) that is safe for A (保证不形成回路)')
        print('       A <- A ∪ {(u, v)}')
        print('  return A')
        print('')
        print('识别安全边的一条规则：')
        print('定理23.1 设图G=(V,E)是一个无向连通图，并且在E上定义了一个具有实数值的加权函数w.',
            '设A是E的一个子集，它包含于G的某个最小生成树中.',
            '设割(S,V-S)是G的任意一个不妨害A的割,且边(u,v)是通过集合A来说是安全的')
        print('推论23.2 设G=(V,E)是一个无向连通图,并且在E上定义了相应了实数值加权函数w',
            '设A是E的子集，且包含于G的某一最小生成树中。设C=(Vc,Ec)为森林GA=(V,A)的一个连通分支(树)',
            '如果边(u,v)是连接C和GA中其他某联通分支的一条轻边,则(u,v)对集合A来说是安全的')
        print('证明:因为割(Vc,V-Vc)不妨害A，(u,v)是该割的一条轻边。因此(u,v)对A来说是安全的')
        print('练习23.1-1 设(u,v)是图G中的最小权边.证明:(u,v)属于G的某一棵最小生成树')
        print('练习23.1-2 略')
        print('练习23.1-3 证明：如果一条边(u,v)被包含在某一最小生成树中,那么它就是通过图的某个割的轻边')
        print('练习23.1-4 因为这条边虽然是轻边，但是连接后产生不安全的回路')
        print('练习23.1-5 设e是图G=(V,E)的某个回路上一条最大权边.证明：存在着G\'=(V,E-{e})的一棵最小生成树,',
            '它也是G的最小生成树。亦即，存在着G的不包含e的最小生成树')
        print('练习23.1-6 证明：一个图有唯一的最小生成树,如果对于该图的每一个割,都存在着通过该割的唯一一条轻边',
            '但是其逆命题不成立')
        print('练习23.1-7 论证：如果图中所有边的权值都是正的，那么，任何连接所有顶点、',
            '且有着最小总权值的边的子集必为一棵树')
        print('练习23.1-8 设T是图G的一棵最小生成树，L是T中各边权值的一个已排序的列表',
            '证明：对于G的任何其他最小生成树T\'，L也是T\'中各边权值的一个已排序的列表')
        print('练习23.1-9 设T是图G=(V,E)的一棵最小生成树,V\'是V的一个子集。设T\'为T的一个基于V\'的子图',
            'G\'为G的一个基于V\'的子图。证明:如果T\'是连通的,则T\'是G\'的一棵最小生成树')
        print('练习23.1-10 给定一个图G和一棵最小生成树T,假定减小了T中某一边的权值。',
            '证明：T仍然是G的一棵最小生成树。更形式地,设T是G的一棵最小生成树',
            '其各边的权值由权值函数w给出.')
        print(' 证明：T是G的一棵最小生成树，其各边的权值由w\'给出')
        print('练习23.1-11 给定一个图G和一棵最小生成树T，假定减小了不在T中的某条边的权值',
            '请给出一个算法,来寻找经过修改的图中的最小生成树')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

class Chapter23_2:
    '''
    chpater23.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter23.2 note

        Example
        ====
        ```python
        Chapter23_2().note()
        ```
        '''
        print('chapter23.2 note as follow')
        print('23.2 Kruskai算法和Prim算法')
        print('本节所介绍的两种最小生成树算法是对上一节介绍的通用算法的细化')
        print('均采用了一个特定的规则来确定GENERIC-MST算法所描述的安全边')
        print(' 在Kruskal算法中,集合A是一个森林,加入集合A中的安全边总是图中连接两个不同连通分支的最小权边')
        print(' 在Prim算法中,集合A仅形成单棵树,',
            '添加入集合A的安全边总是连接树与一个不在树中的顶点的最小权边')
        print('Kruskal算法')
        print(' 该算法找出森林中连接任意两棵树的所有边中,具有最小权值的边(u,v)作为安全边',
            '并把它添加到正在生长的森林中')
        print(' 设C1和C2表示边(u,v)连接的两棵树,因为(u,v)必是连接C1和其他某棵树的一条轻边',
            '所以由推论23.2可知,(u,v)对C1来说是安全边。Kruskal算法同时也是一种贪心算法',
            '因为在算法的每一步中,添加到森林中的边的权值都是尽可能小的')
        _mst.test_mst_kruskal()
        print(' Kruskal算法在图G=(V,E)上的运行时间取决于不相交集合数据结构是如何实现的')
        print(' 使用*按秩结合*和*路径压缩*的启发式方法实现不相交集合森林，从渐进意义上来说是最快的方法')
        print(' 综上所述：算法总的运行时间所需时间为O(ElgE),由于E<V^2',
            '因而lgE=O(lgV),于是也可以将Kruskal算法的运行时间重新表述为O(ElgV)')
        print('Prism算法')
        print(' 如Kruskal算法一样,Prism算法也是通用最小生成树算法的特例')
        print(' Prism算法的执行非常类似于寻找图的最短路径的Dijkstra算法')
        print(' Prism算法的特点是集合A中边总形成单棵树')
        print(' 树从任意根顶点r开始形成,并逐渐形成,直至该树覆盖了V中所有的顶点')
        print(' 在每一步，一条连接了树A与GA=(V,A)中某孤立顶点的轻边被加入到树A中')
        print(' 由推论23.2可知，该规则仅加入对A安全的边，因此当算法终止时,',
            'A中的边就形成了一棵最小生成树')
        print(' 因为每次添加到树中的边都是使树的权尽可能的小的边.因此策略也是贪心的')
        print('有效实现Prism算法的关键是设法较容易地选择一条新的边,将其添加到由A的边所形成的树中',
            '算法的输入是连通图G和待生成的最小生成树根r')
        print('在算法的执行过程中,不在树中的所有顶点都放在一个基于key域的最小优先级队列Q中')
        print('对每个顶点v来说,key[v]是所有将v与树中某一顶点相连的边中的最小权值;')
        print('按据约定,如果不存在这样的边,则key[v]=∞,pi[v]=None')
        print('当算法终止时,最小优先队列Q是空的,而G的最小生成树A则满足：')
        print(' A={(v,pi[v]):v∈V-{r}}')
        print('Prism算法的性能,取决于优先队列Q是如何实现的,如果用二叉最小堆实现,其运行时间为O(V)',
            '由于EXTRACT-MIN操作需要O(lgV)时间,所以对EXTRACT-MIN的全部调用所占用的之间为O(VlgV)')
        print('通过使用斐波那契堆,Prism的算法渐进运行时间可得到进一步改善,',
            '可在O(lgV)的平摊时间内完成EXTRACT-MIN操作,在O(1)的平摊时间里完成DECRESE-KEY操作')
        print('因此,如果使用斐波那契堆来实现最小优先队列Q,Prism算法的运行时间可以改进为O(E+VlgV)')
        print('练习23.2-1 根据对边进行排序不同，即使对同一输入图,Kruskal算法也可能得出不同的生成树',
            '证明对G的每一棵最小生成树T,Kruskal算法中都存在一种方法来对边进行排序,使得算法返回的最小生成树为T')
        print('练习23.2-2 假定图G=(V,E)用邻接矩阵表示,在这种条件下,给出Prism算法的运行时间为O(V^2)的实现')
        print('练习23.2-3 稀疏图G=(V,E),|E|=Θ(V),稠密图|E|=Θ(V^2)')
        print('练习23.2-4 因为已经知道了权值的上限，采用计数排序进行权重排序加速')
        print('练习23.2-5 使用斐波那契堆进行加速排序')
        print('练习23.2-6 使用桶排序进行加速排序')
        print('练习23.2-7 假设某个图G有一棵已经计算出来的最小生成树。',
            '如果一个新的顶点以及关联的边被加入到了G中,该最小生成树可以多块的时间内被更新')
        print('练习23.2-8 分治算法计算最小生成树，给定一个图G=(V,E),将顶点集合V划分成两个集合V1和V2',
            '使得|V1|和|V2|至多差1.设E1为一个边集,其中的边都与V1中的顶点关联',
            'E2为另一个边集,其中的边都与V2中的顶点关联.在两个子图G1=(V1,E1)和G2=(V2,E2)上,',
            '分别递归地解决最小生成树问题.最后，从E中选择一条通过割集(V1,V2)的最小权边,',
            '并利用该边,将所得的两棵最小生成树合并成一棵完整的生成树')
        print('思考题23-1 次最优的最小生成树')
        print(' a)证明最小生成树是唯一的,但次最优最小生成树未必一定是唯一的')
        print(' b)设T是G的一棵最小生成树,证明存在边(u,v)∈T和(x,y)∉T,',
            '使得T-{(u,v)}∪{(x,y)}是G的一棵次最优最小生成树')
        print(' c)设T是G的一棵生成树,且对任意两个顶点u,v∈V,设max[u,v]是T中u和v',
            '之间唯一通路上的具有最大权值的边.请给出一个运行时间为O(V^2)的算法,',
            '在给定T和所有顶点u,v∈V以后,可以计算出max[u,v]')
        print(' 最小生成树也是无向无环连通图')
        print('思考题23-2 稀疏图的最小生成树')
        print(' 对于一个非常稀疏的连通图G=(V,E),可以对G进行预处理,以便在运行Prism算法前减少结点的数目',
            '这样对使用了斐波那契堆的Prism算法,就能对其原有的运行时间O(E+VlgV)做进一步的改进')
        print(' 特别地,对于每个结点u,都选择与u关联的最小权边(u,v),并将(u,v)添加到正在构造的最小生成树中',
            '接着,收缩所有被选中的边')
        print(' 接着,在搜索这些边时,不是一次收缩一条,而是首先找出被组合成同一个新结点的那组结点')
        print('原图中的若干条边可能会被重命名成相同的名称.')
        print('思考题23-3 瓶颈生成树')
        print(' 瓶颈生成树最大的边权值在G的所有生成树中是最小的。',
            '瓶颈生成树的值为T中最大权值边的权')
        print(' a) 论证:最小生成树也是瓶颈生成树')
        print('思考题23-4 其他最小生成树算法 略')
        # python src/chapter23/chapter23note.py
        # python3 src/chapter23/chapter23note.py

chapter23_1 = Chapter23_1()
chapter23_2 = Chapter23_2()

def printchapter23note():
    '''
    print chapter23 note.
    '''
    print('Run main : single chapter twenty-three!')  
    chapter23_1.note()
    chapter23_2.note()

# python src/chapter23/chapter23note.py
# python3 src/chapter23/chapter23note.py
if __name__ == '__main__':  
    printchapter23note()
else:
    pass

```

```py

import graph as _g
import notintersectset as _s
import math as _math
from copy import deepcopy as _deepcopy

class _MST:
    def __init__(self, *args, **kwwords):
        pass

    def generic_mst(self, g: _g.Graph):
        '''
        通用最小生成树算法

        Args
        ===
        `g` : 图G=(V,E)

        `w` : 图的权重

        Doc
        ===
        # A = []

        # while A does not form a spanning tree

        #    do find an edge (u,v) that is safe for A (保证不形成回路)

        #       A <- A ∪ {(u, v)}

        # return A

        '''
        A = ['generic mst']
        g.edges.sort()
        return A

    def mst_kruskal(self, g : _g.Graph):
        '''
        最小生成树的Kruska算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `(mst_list, weight)` : 最小生成树列表和最小权重组成的`tuple`

        Example
        ===
        ```python
        g = Graph()
        g.clear()
        g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
        g.addedgewithweight('a', 'b', 4)
        g.addedgewithweight('b', 'c', 8)
        g.addedgewithweight('c', 'd', 7)
        g.addedgewithweight('d', 'e', 9)
        g.addedgewithweight('a', 'h', 8)
        g.addedgewithweight('b', 'h', 11)
        g.addedgewithweight('c', 'i', 2)
        g.addedgewithweight('i', 'h', 7)
        g.addedgewithweight('h', 'g', 1)
        g.addedgewithweight('g', 'f', 2)
        g.addedgewithweight('f', 'e', 10)
        g.addedgewithweight('d', 'f', 14)
        g.addedgewithweight('c', 'f', 4)
        g.addedgewithweight('i', 'g', 4)
        mst_kruskal(g)
        >>> ([('h', 'g', 1), ('c', 'i', 2), ('g', 'f', 2), ('a', 'b', 4), ('c', 'f', 4), ('c', 'd', 7), ('b', 'c', 8), ('d', 'e', 9)], 37)
        ```
        '''
        s = _s.Set()
        A = []
        weight = 0
        for v in g.veterxs:
            s.make_set(v)
        g.edges.sort()
        for e in g.edges:
            (u, v) = g.getvertexfromedge(e)
            uset = s.find(u)
            vset = s.find(v)
            if uset != vset:
                A += [(u.key, v.key, e.weight)]
                s.union(uset, vset)
                weight += e.weight
        return A, weight

    def __change_weightkey_in_queue(self, Q, v, u):
        for q in Q:
            if q.key == v.key:
                q.weightkey = v.weightkey
                q.pi = u
                break

    def mst_prism(self, g : _g.Graph, r : _g.Vertex):
        '''
        最小生成树的Prism算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `weight` : 最小权重
        '''
        for u in g.veterxs:
            u.isvisit = False
            u.weightkey = _math.inf
            u.pi = None
        if type(r) is not _g.Vertex:
            r = g.veterxs_atkey(r)
        else:
            r = g.veterxs_atkey(r.key)
        r.weightkey = 0   
        total_adj = g.getadj_from_matrix()
        weight = 0
        n = g.vertex_num
        weight_min = 0
        k = 0
        tree = []
        for j in range(n):
            weight_min = _math.inf
            u = None
            # 优先队列Q extract-min
            for v in g.veterxs:
                if v.isvisit == False and v.weightkey < weight_min:
                    weight_min = v.weightkey
                    u = v
            u.isvisit = True
            # 获取u的邻接表
            adj = g.getvertexadj(u)
            # 计算最小权重
            weight += weight_min        
            for v in adj:
                # 获取边
                edge = g.getedge(u, v)
                # 构造最小生成树
                if weight_min != 0 and edge.weight == weight_min:
                    tree.append((v.key, u.key, weight_min))
                # if v ∈ Q and w(u, v) < key[v]
                if v.isvisit == False and edge.weight < v.weightkey:
                    v.pi = u
                    v.weightkey = edge.weight
                    # 更新Vertex域 如果是引用则不需要，此处adj不是引用
                    for q in g.veterxs:
                        if q.key == v.key:
                            q.weightkey = v.weightkey
                            q.pi = v.pi
                            break
        return tree, weight

    def mst_dijkstra(self, g: _g.Graph, r: _g.Vertex):
        '''
        最小生成树的Prism算法 时间复杂度`O(ElgV)`
        Args
        ===
        `g` : 图`G=(V,E)`

        Return
        ===
        `weight` : 最小权重
        '''
        for u in g.veterxs:
            u.isvisit = False
            u.weightkey = _math.inf
            u.pi = None
        if type(r) is not _g.Vertex:
            r = g.veterxs_atkey(r)
        else:
            r = g.veterxs_atkey(r.key)
        r.weightkey = 0
        total_adj = g.getadj_from_matrix()
        weight = 0
        n = g.vertex_num
        weight_min = 0
        k = 0
        tree = []
        for j in range(n):
            weight_min = _math.inf
            u = None
            # 优先队列Q extract-min
            for v in g.veterxs:
                if v.isvisit == False and v.weightkey < weight_min:
                    weight_min = v.weightkey
                    u = v
            u.isvisit = True
            # 获取u的邻接表
            adj = g.getvertexadj(u)
            # 计算最小权重
            weight += weight_min
            for v in adj:
                # 获取边
                edge = g.getedge(u, v)
                # 构造最小生成树
                if weight_min != 0 and edge.weight == weight_min:
                    tree.append((v.key, u.key, weight_min))
                # if v ∈ Q and w(u, v) < key[v]
                if v.isvisit == False and edge.weight < v.weightkey:
                    v.pi = u
                    v.weightkey = edge.weight
                    # 更新Vertex域 如果是引用则不需要，此处adj不是引用
                    for q in g.veterxs:
                        if q.key == v.key:
                            q.weightkey = v.weightkey
                            q.pi = v.pi
                            break
        return tree, weight

__mst_instance = _MST()
generic_mst = __mst_instance.generic_mst
mst_kruskal = __mst_instance.mst_kruskal
mst_prism = __mst_instance.mst_prism
mst_dijkstra = __mst_instance.mst_dijkstra

def buildgraph():
    '''
    构造图
    '''
    g =  _g.Graph()
    g.clear()
    g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    g.addedgewithweight('a', 'h', 8)
    g.addedgewithweight('a', 'b', 4)
    g.addedgewithweight('b', 'c', 8)
    g.addedgewithweight('c', 'd', 7)
    g.addedgewithweight('d', 'e', 9)
    g.addedgewithweight('b', 'h', 11)
    g.addedgewithweight('c', 'i', 2)
    g.addedgewithweight('i', 'h', 7)
    g.addedgewithweight('h', 'g', 1)
    g.addedgewithweight('g', 'f', 2)
    g.addedgewithweight('f', 'e', 10)
    g.addedgewithweight('d', 'f', 14)
    g.addedgewithweight('c', 'f', 4)
    g.addedgewithweight('i', 'g', 6)
    return g

def test_mst_generic():
    g = _g.Graph()
    g.clear()
    g.addvertex(['a', 'b', 'c', 'd'])
    g.addedgewithweight('a', 'b', 2)
    g.addedgewithweight('a', 'd', 3)
    g.addedgewithweight('b', 'c', 1)
    print('邻接表为')
    _g._print_inner_conllection(g.adj)
    print('邻接矩阵为')
    print(g.matrix)
    print('图G=(V,E)的集合为')
    _g._print_inner_conllection(g.edges)
    print(generic_mst(g))
    print('边按权重排序后图G=(V,E)的集合为')
    _g._print_inner_conllection(g.edges)
    del g

def test_mst_kruskal():
    g = buildgraph()
    print('边和顶点的数量分别为:', g.edge_num, g.vertex_num)
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为')
    print(g.matrix)
    print('最小生成树为：')
    mst_list = mst_kruskal(g)
    print(mst_list)
    del g

def test_mst_prism():
    g = buildgraph()
    print('边和顶点的数量分别为:', g.edge_num, g.vertex_num)
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为')
    print(g.matrix)
    print('最小生成树为：')
    mst_list = mst_prism(g, 'a')
    print(mst_list)
    del g

def test_mst_dijkstra():
    g = buildgraph()
    print('邻接表为')
    g.printadj()
    print('邻接矩阵为：')
    print(g.matrix)
    del g

def test():
    '''
    测试函数
    '''
    test_mst_generic()
    test_mst_kruskal()
    test_mst_prism()
    test_mst_dijkstra()

if __name__ == '__main__':
    print('test as follows')
    test()
else:
    pass
```

```py

class UndirectedGraph:
    '''
    无向图 `G=(V, E)`
    '''
    def __init__(self, vertexs : list = [], edges : list = []):
        '''
        无向图 `G=(V, E)`

        Args
        ===
        `vertexs` : 顶点集合 `list` contains element which contains one element denote a point

        `edges` : 边集合 `list` contains element which contains two elements denote one edge of two points repectively

        Example
        ===
        ```python
        import notintersectset as graph
        >>> g = graph.UndirectedGraph(['a', 'b', 'c', 'd'], [('a', 'b')])
        ```
        '''
        self.vertexs = vertexs
        self.edges = edges
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0

    def get_connected_components(self):
        '''
        获取无向图中连通子图的集合
        '''
        self.__findcount = 0
        self.__unioncount = 0
        self.__kcount = 0
        set = Set()
        for v in self.vertexs:
            set.make_set(v)
        for e in self.edges:
            u, v = e
            set1 = set.find(u)
            set2 = set.find(v)
            self.__findcount += 2
            if set1 != set2:
                set.union(set1, set2)
                self.__unioncount += 1
        self.__kcount = len(set.sets)
        return set

    def print_last_connected_count(self):
        '''
        获取上一次连接无向图之后调用函数情况
        '''
        print('the k num:{} the find num:{} the union num:{}'. \
            format(self.__kcount, self.__findcount, self.__unioncount))

class Set:
    '''
    不相交集合数据结构
    '''
    def __init__(self):
        '''
        不相交集合数据结构
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        self.sets.append({element})

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并, 或者`set1`和`set2`所代表的集合合并
        '''
        if set1 is None or set2 is None:
            return
        if type(set1) is not set:
            set1 = self.find(set1)
        if type(set2) is not set:
            set2 = self.find(set2)
        self.sets.remove(set1)
        self.sets.remove(set2)
        self.sets.append(set1 | set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            if element in set:
                return set
        return None
    
    def __str__(self):
        return str(self.sets)

    def printsets(self):
        '''
        打印集合
        '''
        for set in self.sets:
            print(set)

class ListNode:
    def __init__(self, key = None):
        '''
        采用链表表示不相交集合结点
        '''
        self.first = None
        self.next = None
        self.key = key
    
    def __str__(self):
        return str(self.key)

class List:
    def __init__(self):
        '''
        采用链表表示不相交集合
        '''
        self.rep = None
        self.head = None
        self.tail = None
        self.size = 0
    
    def __str__(self):
        return 'List size:{} and rep:{}'.format(self.size, self.rep)

class ListSet(Set):
    '''
    不相交集合的链表表示
    '''
    def __init__(self):
        '''
        不相交集合的链表表示
        '''
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        list = List()
        node = ListNode(element)  
        if list.size == 0:           
            list.head = node
            list.tail = node
            list.rep = node
            node.first = node
            list.size = 1
        else:
            list.tail.next = node
            list.tail = node 
            node.first = list.head
            list.size += 1
        self.sets.append(list)

    def union(self, set1, set2):
        '''
        将子集合`set1`和`set2`合并
        '''
        self.sets.remove(set1)
        self.sets.remove(set2)
        set1.tail.next = set2.rep
        set1.size += set2.size
        set1.tail = set2.tail

        set2.rep = set1.rep

        node = set2.head
        for i in range(set2.size):
            node.first = set1.rep
            node = node.next

        self.sets.append(set1)

    def unionelement(self, element1, element2):
        '''
        将`element1`代表的集合和`element2`代表的集合合并
        '''
        set1 = self.find(element1)
        set2 = self.find(element2)
        if set1 is None or set2 is None:
            return
        if set1.size < set2.size:
            self.union(set2, set1)
        else:
            self.union(set1, set2)

    def find(self, element):
        '''
        找出包含元素`element`的集合
        '''
        for set in self.sets:
            node = set.rep
            while node != set.tail:
                if node.key == element:
                    return set
                node = node.next
            else:
                if set.tail.key == element: 
                    return set
        return None

class RootTreeNode:
    '''
    有根树结点
    '''
    def __init__(self, key = None, parent = None, rank = None):
        '''
        有根树结点

        Args
        ===
        `key` : 关键字值

        `parent` : 结点的父结点

        `rank` : 结点的秩
        '''
        self.key = key
        self.parent = parent
        self.rank = rank
    
    def __str__(self):
        return 'key:{} rank:{}'.format(self.key, self.rank)

class RootTree:
    '''
    有根树
    '''
    def __init__(self, root = None):
        '''
        有根树
        Args
        ===
        `root` : 有根树的根结点
        '''
        self.root = root
    
    def __str__(self):
        return 'roots:' + str(self.root)

class ForestSet(Set):
    '''
    不相交集合森林
    '''
    def __init__(self):
        self.sets = []

    def make_set(self, element):
        '''
        用元素`element`建立一个新的集合
        '''
        treenode = RootTreeNode(element)
        self.make_set_node(treenode)
        
    def make_set_node(self, node : RootTreeNode):
        '''
        用有根树结点`node`建立一个新的集合
        '''
        tree = RootTree()  
        node.parent = node
        node.rank = 0
        tree.root = node
        self.sets.append(tree)

    @classmethod
    def link(self, node1 : RootTreeNode, node2 : RootTreeNode):
        '''
        连接两个有根树的结点`node1`和`node2`
        '''
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1
    
    def union(self, x : RootTreeNode, y : RootTreeNode):
        '''
        将有根树结点`x`代表的集合和有根树结点`y`代表的集合合并
        '''
        self.link(self.findnode(x), self.findnode(y))

    def findnode(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合
        '''
        if x != x.parent:
            x.parent = self.findnode(x.parent)
        return x.parent

    def findnode_nonrecursive(self, x : RootTreeNode):
        '''
        带路径压缩的寻找集合(非递归版本)
        '''
        y = x
        while y != y.parent:
            y = y.parent
        while x != x.parent:
            x.parent = y
            x = x.parent

def connected_components(g: UndirectedGraph):
    '''
    求一个无向图中连通子图的个数

    Args
    ===
    `g` : UndirectedGraph 无向图

    '''
    set = Set()
    for v in g.vertexs:
        set.make_set(v)
    for e in g.edges:
        u, v = e
        set1 = set.find(u)
        set2 = set.find(v)
        if set1 != set2:
            set.union(set1, set2)
    return set

def test_graph_connected():
    '''
    测试无向图链接连通子图
    '''
    g = UndirectedGraph()
    g.vertexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    g.edges.append(('b', 'd'))
    g.edges.append(('e', 'g'))
    g.edges.append(('a', 'c'))
    g.edges.append(('h', 'i'))
    g.edges.append(('a', 'b'))
    g.edges.append(('e', 'f'))
    g.edges.append(('b', 'c'))
    print(g.get_connected_components())
    g.print_last_connected_count()

def test_list_set():
    '''
    不相交集合的链表表示
    '''
    NUM = 16
    set = ListSet()
    for i in range(NUM):
        set.make_set(i)
    for i in range(0, NUM - 1, 2):
        set.unionelement(i, i + 1)
    for i in range(0, NUM - 3, 4):
        set.unionelement(i, i + 2)
    set.printsets()
    set.unionelement(1, 5)
    set.unionelement(11, 13)
    set.unionelement(1, 10)
    set.printsets()
    print(set.find(2))
    print(set.find(9))

def test_forest_set():
    '''
    测试不相交集合森林
    '''
    NUM = 16
    set = ForestSet()
    nodes = []
    for i in range(NUM):
        nodes.append(RootTreeNode(i))
    for i in range(NUM):
        set.make_set_node(nodes[i])
    set.printsets()
    for i in range(0, NUM - 1, 2):
        set.union(nodes[i], nodes[i + 1])
    set.printsets()
    for i in range(0, NUM - 3, 4):
        set.union(nodes[i], nodes[i + 2])
    set.printsets()

if __name__ == '__main__':
    test_graph_connected()
    test_list_set()
    test_forest_set()
else:
    pass    

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter24/chapter24note.py
# python3 src/chapter24/chapter24note.py
'''

Class Chapter24_1

Class Chapter24_2

Class Chapter24_3

Class Chapter24_4

Class Chapter24_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    import graph as _g
    import shortestpath as _sp
else:
    from . import graph as _g
    from . import shortestpath as _sp

class Chapter24_1:
    '''
    chpater24.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.1 note

        Example
        ====
        ```python
        Chapter24_1().note()
        ```
        '''
        print('chapter24.1 note as follow')  
        print('第24章 单源最短路径')
        print('一种求最短路径的方式就是枚举出所有从芝加哥到波士顿的路线,',
            '并对每条路线的长度求和,然后选择最短的一条')
        print('在最短路径问题中,给出的是一个带权有向图G=(V,E),加权函数w:E->R为从边到实型权值的映射')
        print('路径p=<v0,v1,...,vk>的权是指其组成边的所有权值之和')    
        print('边的权值还可以被解释为其他的某种度量标准,而不一定是距离')
        print('它常常被用来表示时间,费用,罚款,损失或者任何其他沿一',
            '条路径线性积累的试图将其最小化的某个量')
        # !广度优先搜索算法就是一种在无权图上执行的最短路径算法
        print('广度优先搜索算法就是一种在无权图上执行的最短路径算法,',
            '即在图的边都具有单位权值的图上的一种算法')
        print('单源最短路径的变体')
        print(' 已知图G=(V,E),希望找出从某给定源顶点s∈V到每个顶点v∈V的最短路径。')
        print('很多其他问题都可用单源问题的算法来解决,其中包括下列变体')
        print(' 1.单终点最短路径问题:找出从每个顶点v到指定终点t的最短路径')
        print(' 2.单对顶点最短路径问题:对于某给定顶点u和v,找出从u和v的一条最短路径')
        print('   如果解决了源点为u的单源问题,则这一问题也就获得解决')
        print(' 3.对于每对顶点u和v,找出从u到v的最短路径')
        print('   虽然将每个顶点作为源点,运行一次单源算法就可以解决这一问题,但通常可以更快地解决这一问题')
        print('最短路径的最优子结构')
        print('  最短路径算法通常依赖于一种性质,也就是一条两顶点间的最短路',
            '径包含路径上其他的最短路径,这种最优子结构性质是动态规划和贪心算法是否适用的一个标记')
        print('Dijkstra算法是一个贪心算法,而找出所有顶点对之间的最短路径的',
            'Floyd-Warshall算法是一个动态规划算法')
        print('引理24.1(最短路径的子路径是最短路径)对于一给定的带权有向图G=(V,E),所定义的权函数为w',
            'E->R。设p=<v1,v2,..,vk>是从v1到vk的最短路径')
        print('负权值边')
        print(' 在单源最短路径问题的某些实例中,可能存在着权值为负值的边.',
            '如果图G=(V,E)不包含从源s可达的负权回路,则对所有v∈V,最短路径的权的定义d(u,v)依然正确,',
            '即使它是一个负值也是如此.但是,如果存在一条从s可达的负权回路,那么最短路径的权的定义就不能成立')
        print('从s到该回路上的顶点之间就不存在最短路径,因为我们总是可以顺着已找出的\"最短\"路径,',
            '再穿过负权值回路而获得一条权值更小的路径.',
            '因此,如果从s到v的某路径中存在一条负权回路,就定义d(u,v)=-inf')
        print('一些最短路径算法,如Dijstra算法,假定输入图中的所有边的权值都是非负的,如公路地图的例子',
            '另一些算法,如Bellman-Ford算法,允许输入图中存在负权边,只要不存在从源点可达的负权回路')
        print('特别地,如果存在负权回路,算法还可以检测并报告这种回路的存在')
        print('一条最短路径能包含回路嘛?不能包含负权回路.也不会包含正权回路,因为从路径上移去回路后,',
            '可以产生一个具有相同源点和终点、权值更小的路径')
        print('最短路径的表示')
        print('不仅希望算出最短路径的权,而且也希望得到最短路径设置pi属性,',
            '以便使源于顶点v的前辈链表沿着从s到v的最短路径的相反方向排列')
        print('对于一给定的v.pi=None的顶点v,',
            '可以运用PRINT-PATH(G,s,v)输出从s到v的一条最短路径')
        print('不过,在最短路径算法的执行过程中,无需用pi的值来指明最短路径。',
            '正如广度优先搜索一样,是由pi值导出的前趋子图Gpi=(Vpi,Epi).这里,',
            '定义顶点集Vpi为G中所有具有非空前趋的顶点集合,再加上源点s')
        # !最短路径并不一定是唯一的,最短路径树亦是如此
        print('最短路径并不一定是唯一的,最短路径树亦是如此')
        print('松弛技术')
        print(' 本章的算法用到了松弛(relaxation)技术.',
            '对每个顶点v∈V,都设置一个属性d[v],用来描述从源点s到v',
            '的最短路径上权值的上界,称为最短路径估计')
        print('在松弛一条边(u,v)的过程中,要测试是否可以通过u,对找到的到v的最短路径进行改进',
            '如果可以改进的话,则更新d[v]和pi[v].一次松弛操作可以减小最短路径估计的值d[v]',
            '并更新v的前趋pi[v]')
        print('本章每个算法都会调用INITIALIZE-SINGLE-SOURCE,然后重复对边进行松弛的过程',
            '另外,松弛是改变最短路径和前趋的唯一方式')
        print('本章中的算法之间的区别在于对每条边进行松弛操作的次数',
            '以及对边执行松弛操作的次序有所不同')
        print('在Dijkstra算法以及关于有向无回路图的最短路径算法中,对每条边执行一次松弛操作')
        print('在Bellman-Ford算法中,对每条边要执行多次松弛操作')
        print('最短路径以及松弛的性质(隐含地假设了图是调用INITIALIZE-SINGLE-SOURCE(G,s)进行初始化的,'
            '且最短路径估计和前趋子图唯一的变化途径就是一系列的松弛步骤')
        print('1.三角不等式(引理24.10)')
        print('  对任意边(u,v)∈E,有d(s,v)<=d(s,u)+w(u,v)')
        print('2.上界性质(引理24.11)')
        print('  对任意顶点v∈V,有d[v]>=d(s,v),而且一旦d[v]达到d(s,v)值就不再改变')
        print('3.无路径性质(推论24.12)')
        print('  如果从s到v不存在路径,则总是有d[v]=d(s,v)=inf')
        print('4.收敛性质(引理24.14)')
        print('  如果s-u->v是图G某u,v∈V的最短路径,而且在松弛边(u,v)',
            '之前的任何时间d[u]=d(s,u),则在操作之后总有d[v]=d(s,v)')
        print('路径松弛性质(引理24.15)')
        print('这个性质的保持并不受其他松弛操作的影响,',
            '即使它们与p的边上的松弛操作混合在一起也是一样的')
        print('前趋子图性质(引理24.17)')
        print('  一旦对于所有v∈V,d[v]=d(s,v),前趋子图就是一个以s为根的最短路径树')
        print('Bellman-Ford算法,该算法用来解决一般(边的权值可以为负)的单源最短路径问题')
        print('Bellman-Ford算法非常简单,可以检测是否有从源点可达的负权回路')
        print('在一个有向无环图中,在线性时间内计算出单源最短路径的算法')
        print('Dijkstra算法,它的运行时间比Bellman-Ford算法低,但要求所有边的权值为非负')
        print('使用Bellman-Ford算法来解决\"动态规划\"的一个特例')
        print('24.5节证明了上面所陈述的最短路径和松弛的性质')
        print('所有算法都假设有向图G用邻接表的形式存储,而且每条边上还存储了它的权值')
        print('当遍历每一个邻接表时,可以对每条边在O(1)时间内确定其权值')
        print('24.1 Bellman-Ford算法')
        print('Bellmax-Ford算法能在一般的情况下(存在负边权的情况)下,解决单源最短路径问题',
            '对于给定的带权有向图G=(V,E),其源点为s,加权函数为w')
        print('Bellman-Ford算法后可以返回一个布尔值,表明图中是否存在着一个从源点可达的权为负的回路')
        print('若存在这样的回路的话,算法说明该问题无解;若不存在这样的回路,算法将产生最短路径及其权值')
        print('算法运用松弛技术,对每个顶点v∈V,逐步减小源s到v的最短路径的权的估计值d[v]直至其达到的实际最短路径的权d(s,v)')
        print('算法返回布尔值TRUE,当且仅当图中不包含从源点可达的负权回路')
        print('引理24.2 设G=(V,E)为带权有向图,其源点为s,权函数为w:E->R,',
            '并且假定G中不包含从s点可达的负权回路')
        print('推论24.3 设G=(V,E)为带权有向图,源顶点为s,加权函数为w,E->R.',
            '对每一顶点v∈V,从s到v存在一条通路','当且仅当对G运行Bellman-Ford算法,算法终止时,有d[v]<∞')
        print('定理24.4(Bellman-Ford算法的正确性),设G=(V,E)为带权有向图.源点为s,权函数为w：E->R',
            '对该图运行Bellman-Ford算法.若G不包含s可达的负权回路,则算法返回TRUE',
            '对所有顶点v∈V,有d[v]=d(s,v)成立.前趋子图Gpi是以s为根的最短路径树',
            '如果G包含从s可达的负权回路,则算法返回FALSE')
        print('练习24.1-1 以顶点z作为源点,对图24-4所给出的有向图运行Bellman-Ford算法',
            '每趟操作中,按照图中的相同顺序对边进行松弛,并表示出每趟过后d与pi的值',
            '现在,将边(z,x)的权值变为4,再以s为源点运行此算法')
        _sp.test_bellman_ford()
        print('练习24.1-2 证明推论24.3')
        print('练习24.1-3 对于给定的无负权回路的带权有向图G=(V,E),设在所有u,v∈V的顶点对中,',
            'm为所有从u到为v的最短路径上边数最小值中的最大值(这里,最短路径是根据权值来说的,而不是边的数目)',
            '可以对Bellman-Ford算法做简单的修改,则可在m+1趟后终止')
        print('练习24.1-4 对Bellman-Ford算法进行比较,对任意顶点v,',
            '当从源点到v的某些路径上存在一个负权回路,则置d[v]=-∞')
        print('练习24.1-5 设G=(V,E)为一带权有向图,其权函数w:E->R。请给出一个O(VE)时间的算法',
            '对每个顶点v∈V,找出d(v)=min{d(u,v)}')
        print('练习24.1-6 假定一加权有向图G=(V,E)包含一负权回路.',
            '请给出一个能够列出此回路上的顶点的高效算法')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_2:
    '''
    chpater24.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.2 note

        Example
        ====
        ```python
        Chapter24_2().note()
        ```
        '''
        print('chapter24.2 note as follow')
        print('24.2 有向无回路图中的单源最短路径')
        print('按顶点的拓扑序列对某加权dag图(有向无回路图)G=(V,E)的边进行松弛后',
              '就可以在Θ(V+E)时间内计算出单源最短路径.在一个dag图中最短路径总是存在的',
            '因为即使图中有权值为负的边，也不可能存在负权回路')
        print('定理24.5 如果一个带权有向图G=(V,E)有源点s而且无回路',
            '则在DAG-SHORTEST-PATHS终止时,对任意顶点v∈V,有d[v]=d(s,v),',
            '且前趋子图Gpi是最短路径树')
        print('DAG-SHORTEST-PATHS算法一个有趣的应用是在PERT图分析中确定关键路径',
            '在PERT图中,边表示要完成的工作,边的权表示完成特定工作所需时间',
            '如果边(u,v)进入顶点v而边(v,x)离开顶点v,则工作(u,v)必须在工作(v,x)之前完成')
        print('此dag的一个路径表示必须按一定顺序执行工作序列.关键路径是通过dag的一条最长路径,',
            '它对应于执行一个有序的工作序列的最长时间')
        print('关键路径的权值是完成所有工作所需时间的下限')
        print('练习24.2-1 如下')
        _sp.test_dag_shortest_path()
        print('练习24.2-2 略')
        print('练习24.2-3 略')
        print('练习24.2-4 给出一个高效算法统计有向无回路图中的全部路径数')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_3:
    '''
    chpater24.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.3 note

        Example
        ====
        ```python
        Chapter24_3().note()
        ```
        '''
        print('chapter24.3 note as follow')
        print('Dijkstra算法')
        print('Dijkstra算法解决了有向图G=(V,E)上带权的单源最短路径问题,但要求所有边的权值非负',
            ',假定对每条边(u,v)∈E,有w(u,v)>=0')
        print('一个实现的很好的Dijkstra算法比Bellman-Ford算法的运行时间要低')
        print('Dijkstra算法中设置了一顶点集合S,从源点s到集合中的顶点的最终最短路径的权值均确定')
        print('算法反复选择具有最短路径估计的顶点u∈V-S,并将u加入到S中,对u的所有出边进行松弛')
        print('在下面的算法实现中,用到了顶点的最小优先队列Q,排序关键字为顶点的d值')
        # !Dijkstra算法总是在V-S中选择“最轻”或“最近”的顶点插入集合S中,使用了贪心策略
        print('Dijkstra算法总是在V-S中选择“最轻”或“最近”的顶点插入集合S中,使用了贪心策略')
        print('定理24.6 Dijkstra算法的正确性 已知一带权有向图G=(V,E),',
            '其加权函数w的值为非负,源点为s',
            '对该图运行Dijkstra算法,则在算法终止时,对所有u∈V有d[u]=d(s,u)')
        print('推论24.7 已知一加权函数非负且源点为s的带权有向图G=(V,E),若在该图上运行Dijstra算法,',
            '则在算法终止时,前趋子图Gpi是以s为根的最短路径树')
        print('Dijkstra算法的运行时间依赖于最小优先队列的具体实现')
        print(' 利用从1至|V|编好号的顶点，简单地将d[v]存入一个数组的第v项')
        print(' 每一个INSERT和DECREASE-KEY的操作都是O(1)的时间,而每一个EXTRACT-MIN操作为O(V)时间')
        print(' 总计的运行时间为O(V^2+E)=O(V^2)')
        print('特别地,如果是稀疏图的情况,有E=o(V^2/lgV),在这种情况下,',
            '利用二叉最小堆来实现最小优先队列是很有用的')
        print(' 总计的运行时间为O((V+E)lgV)')
        print('从历史的角度看,在Dijstra算法中,DECRESE-KEY的调用比EXTRACT-MIN的调用一般要多的多')
        print('所以任何能够在不增加EXTRACT-MIN操作的平摊时间的同时')
        print('从渐进意义上来说,都能获得比二叉堆更快的实现(比如斐波那契堆)')
        print('Dijkstra算法和广度优先搜索算法以及计算最小生成树的Prim算法都有类似之处')
        print('和广度优先算法的相似性在于,前者的集合S相当于后者的黑色顶点集合')
        print('练习24.3-1 ')
        _sp.test_dijstra()
        print('练习24.3-2 给出一含有负权边的有向图的简单实例,说明Dijkstra算法对其会产生错误的结果')
        print('练习24.3-3 略')
        print('练习24.3-4 已知一有向图G=(V,E),其每条边(u,v)∈E均对应有一个实数值r(u,v)',
            '表示从顶点u到顶点v之间的通信线路的可靠性,取值范围为0<=r(u,v)<=1',
            '定义r(u,v)为从u到v的线路不中断的概率,并假定这些概率是互相独立的')
        print('练习24.3-5 无权有向图G‘运行广度优先搜索,V中顶点被标记成黑色的顺序与DIJKSTRA算法运行于G上时,',
            '从优先队列中删除V中顶点的顺序相同')
        print('练习24.3-6 设G=(V,E)为带权有向图,权函数w：E->{0,1,2...,W},其中W为某非负整数。',
            '修改Dijkstra算法,以使其计算从指定源点s的最短路径所需的运行时间为O(WV+E)(在最小堆算法处加速)')
        print('练习24.3-7 略')
        print('练习24.3-8 假定有一个带权有向图G=(V,E),从源点s出发的边可能有负边,',
            '所有其他的边的权都非负,而且不存在负权回路,论证在这样的图中,Dijkstra算法可以正确地从s找到最短路径')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_4:
    '''
    chpater24.4 note and function
    '''
    def solve_24_4_1(self):
        '''
        求解练习24.4-1
        '''
        g = _g.Graph()
        g.clear()
        vertexs = ['0', '1', '2', '3', '4', '5', '6']
        g.veterxs = vertexs
        g.addedgewithweight('0', '1', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '2', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '3', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '4', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '5', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '6', 0, _g.DIRECTION_TO)
        g.addedgewithweight('1', '2', 1, _g.DIRECTION_FROM)
        g.addedgewithweight('1', '4', -4, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '3', 2, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '5', 7, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '6', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('3', '6', 10, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '2', 2, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '1', -1, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '4', 3, _g.DIRECTION_FROM)
        g.addedgewithweight('6', '3', -8, _g.DIRECTION_FROM)
        print(_sp.bellman_ford(g, vertexs[0]))
        del g
    
    def solve_24_4_2(self):
        '''
        求解练习24.4-2
        '''
        g = _g.Graph()
        g.clear()
        vertexs = ['0', '1', '2', '3', '4', '5']
        g.veterxs = vertexs
        g.addedgewithweight('0', '1', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '2', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '3', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '4', 0, _g.DIRECTION_TO)
        g.addedgewithweight('0', '5', 0, _g.DIRECTION_TO)
        g.addedgewithweight('1', '2', 4, _g.DIRECTION_FROM)
        g.addedgewithweight('1', '5', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('2', '4', -6, _g.DIRECTION_FROM)
        g.addedgewithweight('3', '2', 1, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '1', 3, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '3', 5, _g.DIRECTION_FROM)
        g.addedgewithweight('4', '5', 10, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '3', -4, _g.DIRECTION_FROM)
        g.addedgewithweight('5', '4', -8, _g.DIRECTION_FROM)
        print(_sp.bellman_ford(g, vertexs[0]))

    def note(self):
        '''
        Summary
        ====
        Print chapter24.4 note

        Example
        ====
        ```python
        Chapter24_4().note()
        ```
        '''
        print('chapter24.4 note as follow')
        print('24.4 差分约束与最短路径')
        print('一般的线性规划问题：要对一组线性不等式定义的线性函数进行优化')
        print('简化为寻找单源最短路径的线性规划的一种特殊情形')
        print('由此引出的单源最短路径问题可以运用Bellman-Ford算法来解决，进而解决原线性规划问题')
        print('线性规划')
        print('  一般的线性规划问题中，给定一个m*n的矩阵A,一个m维向量b和一个n维向量c',
            '希望找出由n个元素组成的向量x,在由Ax<=b所给出的m个约束条件下,使目标函数最大')
        print('单纯形法')
        print('  并不总是能在输入规模的多项式时间内运行；',
            '但是还有其他一些线性规划算法是可以以多项式时间运行的')
        print('有时并不关心目标函数，仅仅是希望找出一个可行解,',
            '即一个满足Ax<=b的向量x,或是确定不存在的可行解')
        print('差分约束系统')
        print('  在一个差分约束系统中，线性规划矩阵A的每一行包含一个1和一个-1',
            'A的所有其他元素都为0.因此，由Ax<=b给出的约束条件是m个差分约束集合')
        print('引理24.8 设x=(x1,x2,...,xn)是一个差分约束系统Ax<=b的一个解,d为任意常数',
            '则x+d=(x1+d,x2+d,..,xn+d)也是该系统Ax<=b的解')
        print('  差分约束系统出现在很多不同的应用领域中')
        print('约束图')
        print('  用图形理论观点来解释差分约束系统是很有益的。',
            '在一理想的差分约束系统Ax<=b,m*n的线性规划矩阵A可被看作是n顶点,m条边的图的关联矩阵的转置')
        print('  对于i=1,2,...,n图中每一个顶点vi对应着n个未知量的一个xi.',
            '图中的每个有向边对应着关于两个未知量的m个不等式的其中一个')
        print('更形式地,给定一个差分约束系统Ax<=b，相应的约束图是一个带权有向图G=(V,E),',
            '其中V={v0,v1,...,vn}')
        print('定理24.9 给定一差分系统Ax<=b，设G=(V,E)为其相应的约束图',
            '如果G不包含负权回路，那么x=(d(v0,v1),d(v0,v2),d(v0,v3),...,d(v0,vn))')
        print('是此系统的一可行解。如果G包含负权回路,那么此系统不存在可行解')
        print('差分约束问题的求解')
        print('  由定理24.9知可以采用Bellman-Ford算法对差分约束系统求解')
        print('  在约束图中，从源点v0到其他所有其他顶点均存在边，因此约束图中任何负权回路均从v0可达',
            '如果Bellman-Ford算法返回TRUE，则最短路径给出了此系统的一个可行解',
            '如果Bellman-Ford算法返回FALSE，则差分约束系统无可行解')
        print('关于n个未知量的m个约束条件的一个差分约束系统产生出一个具有n+1顶点和n+m条边的图',
            '因此采用Bell-Ford算法,可以在O((n+1)(n+m)))时间内将系统解决')
        print('可以对算法进行修改，可以使其运行时间变为O(nm),即使m远小于n')
        print('练习24.4-1 对下列差分约束系统找出其可行解,或者说明不存在可行解',
            '由差分约束不等式写出有向带权图，调用Bellman-Ford求解即可',
            '不等式左边x1-x2表示由结点2指向结点1，不等式右边表示边的权')
        self.solve_24_4_1()
        print('练习24.4-2 对下列差分约束系统找出其可行解，或者说明不存在可行解')
        self.solve_24_4_2()
        print('练习24.4-3 在约束图中，从新顶点v0出发的最短路径的权是否可以为正数')
        print('练习24.4-4 试用线性规划方法来表述单对顶点最短路径问题')
        print('练习24.4-5 试说明如何对Bellman-Ford算法稍作修改，',
            '使其在解关于n个未知量的m个不等式所定义的差分约束系统时，运行时间为O(mn)')
        print('练习24.4-6 假定除了差分约束外，还需要处理相等约束',
            '试说明Bellman-Ford算法如何作适当修改,以解决这个约束系统的变形')
        print('练习24.4-7 试说明如何不用附加顶点v0而对约束图运行类Bellman-Ford算法,从而求得差分约束系统的解')
        print('练习24.4-8 设Ax<=b是关于n个未知量的m个约束条件的差分约束系统',
            '证明对其相应的约束图运行Bellman-Ford算法,可以求得满足Ax<=b,并且对所有的xi,有xi<=0')
        print('练习24.4-9 证明Bellman-Ford算法在差分约束系统Ax<=b的约束图上运行时',
            '使(max{xi}-min{xi})取得满足Ax<=b的最小值')
        print('练习24.4-10 假设线性规划Ax<=b中，矩阵A的每一行对应于差分约束条件,即形如xi<=bk或者-xi<=bk的单变量的约束条件')
        print('练习24.4-11 对所有b的元素均为实数，且所有未知量xi必须是整数的情形,写出一个有效算法,以求得差分的约束系统Ax<=b的解')
        print('练习24.4-12 对所有b的元素均为实数且部分(并不一定是全部)未知量xi必须是整数的情形,',
            '写出一个有效算法,以求得差分的约束系统Ax<=b')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

class Chapter24_5:
    '''
    chpater24.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter24.5 note

        Example
        ====
        ```python
        Chapter24_5().note()
        ```
        '''
        print('chapter24.5 note as follow')
        print('论证的几种正确性均依赖于三角不等式、上界性质、',
            '无路径性质、收敛性质、路径松弛性质和前趋子图性质')
        print('引理24.10(三角不等式) 设G=(V,E)为一带权有向图，其权函数w:E->R,源点为s',
            '那么对于所有边(u,v)∈E,有d(s,v)<=d(s,u)+w(u,v)')
        print('对最短路径估计的松弛的效果')
        print('引理24.11(上界性质)设G=(V,E)为有向加权图,其加权函数为w：E->R',
            '设s∈V为源点,INITIALIZE-SINGLE-SOURCE(G,s)对图进行了初始化',
            '那么，对于所有v∈V有d[v]>=d(s,v),而且这个不变式对图G中边的任意系列松弛操作都保持不变',
            '更进一步说,一旦d[v]到达下界d(s,v)将不再改变')
        print('推论24.12(无路径性质)假定在给定的带权有向图G=(V,E)中,权函数为w:E->R',
            '从一源点s∈V到一给定顶点v∈V不存在路径。',
            '那么在INITIALIZE-SINGLE-SOURCE(G,s)对图初始化以后,有d[v]=d(s,v)=∞',
            '在对于G的边进行任意序列的松弛操作后,这个等式作为循环不变式仍然保持')
        print('引理24.13 设G=(V,E)为一个带权有向图,其权函数w:E->R,且(u,v)∈E。',
            '那么,通过执行RELAX(u,v,w)松弛边(u,v)后.有d[v]<=d[u]+w(u,v)')
        print('引理24.14(收敛性质) 设G=(V,E)为一个带权有向图,其权函数为w:E->R,s∈V为一个源点',
            '对某些顶点u,v∈V，设s~u->v为图G中的最短路径。假定G通过调用INITIALIZE-SINGLE-SOURCE(G,s)进行初始化',
            '然后在图G的边上执行了包括调用RELAX(u,v,w)在内的一系列松弛步骤',
            '如果在调用之前d[u]=d(s,u),那么在调用之后的任意时间d[v]=d(s,v)')
        print('引理24.15(路径松弛性质)是G=(V,E)为一带权有向图，权函数w:E->R.s∈V为源点',
            '考虑任意从s=v0到vk的最短路径p=<v0,v1,...,vk>,如果G通过INITIALIZE-SINGLE-SOURCE(G,s)',
            '然后按顺序进行了一系列的松弛步骤,包括松弛边(v0,v1),(v1,v2),...,(vk-1,vk)',
            '那么，经过这些松弛后以及在以后的任意时刻，都有d[vk]=d(s,vk).',
            '无论其他边是否发生松弛(包括与p的边交错地进行的松弛),这一性质都始终保持')
        print('引理24.16 设G=(V,E)为一带权有向图,其权值函数为w:E->R,s∈V为一个源点',
            '并假定G不含从s可达的负权回路，那么，在图INITIALIZE-SINGLE-SOURCE(G,s)初始化后',
            '前趋子图Gpi就构成以s为根的有根树,在对G边任意序列的松弛操作下仍然像不变式一样保持这个性质')
        print('引理24.17(前趋子图性质)设G=(V,E)为一带权有向图,其权函数w:E->R,s∈V为一个源点',
            '而且假定G不含s可达的负权回路.设调用了INITIALIZE-SINGLE-SOURCE(G,s)',
            '然后在G的边上执行了一系列的松弛操作,得到对所有v∈V有d[v]=d(s,v)',
            '因此,前趋子图Gpi是一个以s为根的最短路径树')
        print('练习24.5-1 对图24-2，除图中已画出的两棵树以外,另外再画出两棵图中所示有向图的最短路径树')
        print('练习24.5-2 举出一个带权有向图G=(V,E)的实例，其加权函数为w：E->R,且源点为s,',
            '要求G满足下列性质：对每条边(u,v)∈E，存在包含(u,v)且以s为根的最短路径树',
            '同时存在另一棵以s为根，但不包含(u,v)的最短路径树')
        print('练习24.5-3 略')
        print('练习24.5-4 设G=(V,E)是带权有向图,源点为s,并设G由过程INITIALIZE=SINGLE-SOURCE(G,s)进行了初始化',
            '证明如果经过一系列松弛操作,pi[x]的值被置为非None,则G中包含一个负权回路')
        print('练习24.5-5 设G=(V,E)为带权有向图且不含负权边，设s∈V为源点若v∈V-{s}为从s可达的顶点',
            '则pi[v]是从源s到v的某最短路径中顶点v的前趋,否则pi[v]=None',
            '举出这样的一个图G和给pi赋值的一个例子,说明可以在Gpi中产生回路')
        print('练习24.5-6 设G=(V,E)为带权有向图,其权值函数为w:E->R,且图中不包含负权回路',
            '设s∈V为源点,且G由INITIALIZE-SINGLE-SOURCE(G,s)进行了初始化',
            '证明对每一顶点v∈Vpi,Gpi中存在一条从s到v的通路,且经过任意序列的松弛操作后，这一性质仍然保持')
        print('练习24.5-7 设G=(V,E)为带权有向图且不包含负权回路.设s∈V为源点且G由INITIALIZE-SINGLE-SOURCE(G,s)进行了初始化',
            '证明存在|V|-1步的松弛序列,使得对所有v∈V,d[v]=d(s,v)')
        print('练习24.5-8 设G为任意带权有向图,且存在一源点s可达负权回路。',
            '证明对G的边总可以构造一个无限的松弛序列，使得每个松弛步骤都能对最短路径估计进行修改')
        print('思考题24-1 对Bellman-Ford算法的Yen氏改进')
        print(' 假设对Bellman-Ford算法每一趟中边的松弛顺序作如下安排,在第一趟执行之前',
            '把一任意线性序列v1,v2,...,v|v|赋值给输入图G=(V,E)的各点')
        print(' a)证明对拓扑序列<v1,v2,...,v|v|>,Gf是无回路图；对拓扑序列<v(|v|),v(|v|-1),..,v(1)>')
        print('思考题24-2 嵌套框')
        print(' 如果存在{1,2,...,d}上的某一排列pi,满足xn(1)<y1,xpi(2)<=y2,...,xpi(d)<yd嵌入另一个d维框(y1,y2,..,yd)中')
        print(' a)证明嵌套关系具有传递性')
        print(' b)描述一个有效算法以确定某d维框是否嵌套于另一d维框中')
        print(' c)假定给出一个由n个d维框组成的集合{B1,B2,...,Bn},写出有效算法以找出满足条件Bij嵌入',
            'Bij+1,j=1,2,..,k-1的最长嵌套框序列<Bi1,Bi2,...,Bik>','用变量n和d来描述所给出的算法的运行时间')
        print('思考题24-3 套汇问题')
        print(' 套汇是指利用货币兑率的差异，把一个单位的某种货币转换为大于一个单位的同种货币的方法')
        print('思考题24-4 关于单源最短路径的Gabow定标算法')
        print(' 定标算法对问题进行求解,开始时仅考虑每个相位输入值(例如边的权)的最高位，',
            '接着通过查看最高两位对初始答案进行细微调整,这样逐步查看越来越多的高位信息')
        print('思考题24-5 Karp最小平均权值回路算法')
        print(' 某边回路包含的所有边的平均权值')
        print('思考题24-6 双调最短路径')
        print(' 如果一个序列首先单调递增，然后再单调递减，',
            '或者能够通过循环移位来单调递增再单调递减,这样的序列就是双调的')
        # python src/chapter24/chapter24note.py
        # python3 src/chapter24/chapter24note.py

chapter24_1 = Chapter24_1()
chapter24_2 = Chapter24_2()
chapter24_3 = Chapter24_3()
chapter24_4 = Chapter24_4()
chapter24_5 = Chapter24_5()

def printchapter24note():
    '''
    print chapter24 note.
    '''
    print('Run main : single chapter twenty-four!')  
    chapter24_1.note()
    chapter24_2.note()

# python src/chapter24/chapter24note.py
# python3 src/chapter24/chapter24note.py
if __name__ == '__main__':  
    printchapter24note()
else:
    pass

```

```py

import graph as _g
import math as _math
from copy import deepcopy as _deepcopy

class _ShortestPath:
    '''
    单源最短路径算法集合
    '''
    def __init__(self, *args, **kwords):
        pass

    def initialize_single_source(self, g : _g.Graph, s : _g.Vertex):
        '''
        最短路径估计和前趋进行初始化 时间复杂度Θ(V)
        '''
        for v in g.veterxs:
            v.d = _math.inf
            v.pi = None
        s.d = 0

    def relax(self, u : _g.Vertex, v : _g.Vertex, weight):
        '''
        一步松弛操作
        '''
        if v.d > u.d + weight:
            v.d = u.d + weight
            v.pi = u
    
    def bellman_ford(self, g : _g.Graph, s : _g.Vertex):
        '''
        Bellmax-Ford算法能在一般的情况下(存在负边权的情况)下,解决单源最短路径问题

        时间复杂度 O(VE)

        Args
        ===
        `g` : 图G=(V,E)

        `s` : 源顶点

        Return
        ===
        `exist` : bool 返回一个布尔值,表明图中是否存在着一个从源点可达的权为负的回路
        若存在这样的回路的话,算法说明该问题无解;若不存在这样的回路,算法将产生最短路径以及权值

        `weight` : 权值

        '''
        weight = 0
        if type(s) is not _g.Vertex:
            s = g.veterxs_atkey(s)
        self.initialize_single_source(g, s)
        n = g.vertex_num
        for i in range(n - 1):
            for edge in g.edges:
                u, v = edge.vertex1, edge.vertex2
                u = g.veterxs_atkey(u)
                v = g.veterxs_atkey(v)
                self.relax(u, v, edge.weight)
        for edge in g.edges:
            u, v = edge.vertex1, edge.vertex2
            u = g.veterxs_atkey(u)
            v = g.veterxs_atkey(v)
            if v.d > u.d + edge.weight:
                return False, weight
            weight += edge.weight
        return True, weight
    
    def dag_shortest_path(self, g : _g.Graph, s : _g.Vertex):
        '''
        按顶点的拓扑序列对某加权dag图(有向无回路图)G=(V,E)的边进行松弛后
        就可以在Θ(V+E)时间内计算出单源最短路径.

        Args
        ===
        `g` : 有向无回路图G=(V,E) 

        `s` : 源顶点

        '''
        sort_list = _g.topological_sort(g)
        self.initialize_single_source(g, s)
        for u in sort_list:
            u = g.veterxs_atkey(u)
            adj = g.getvertexadj(u)
            for v in adj:
                edge = g.getedge(u, v)
                self.relax(u, v, edge.weight)
            
    def dijstra(self, g : _g.Graph, s : _g.Vertex):
        '''
        单源最短路径Dijstra算法
        '''
        self.initialize_single_source(g, s)
        S = []
        Q = _deepcopy(g.veterxs)
        while len(Q) != 0:
            Q.sort(reverse=True)
            u = Q.pop()
            S += [u]
            adj = g.getvertexadj(u)
            if adj is not None:
                for v in adj:
                    edge = g.getedge(u, v)
                    self.relax(u, v, edge.weight)

__shortest_path_instance = _ShortestPath()
bellman_ford = __shortest_path_instance.bellman_ford
dag_shortest_path = __shortest_path_instance.dag_shortest_path
dijstra = __shortest_path_instance.dijstra

def test_bellman_ford():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('s'), _g.Vertex('t'), _g.Vertex(
        'x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('s', 't', 6, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'y', 7, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 5, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 8, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', -4, _g.DIRECTION_TO)
    g.addedgewithweight('x', 't', -2, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'x', -3, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', 9, _g.DIRECTION_TO)
    g.addedgewithweight('z', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('z', 's', 2, _g.DIRECTION_TO)
    print(bellman_ford(g, vertexs[0]))
    del g

def test_dag_shortest_path():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dag_shortest_path(g, vertexs[0])
    del g

def test_dijstra():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dijstra(g, vertexs[0])
    del g

def test():
    '''
    测试函数
    '''
    test_bellman_ford()
    test_dag_shortest_path()
    test_dijstra()

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
'''

Class Chapter25_1

Class Chapter25_2

Class Chapter25_3

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    import graph as _g
    import extendshortestpath as _esp
else:
    from . import graph as _g
    from . import extendshortestpath as _esp

class Chapter25_1:
    '''
    chpater25.1 note and function
    '''
    def solve_25_1_1(self):
        g = _g.Graph()
        g.addvertex(['1', '2', '3', '4', '5', '6'])
        g.addedgewithweight('1', '5', -1, _g.DIRECTION_TO)
        g.addedgewithweight('2', '1', 1, _g.DIRECTION_TO)
        g.addedgewithweight('2', '4', 2, _g.DIRECTION_TO)
        g.addedgewithweight('3', '2', 2, _g.DIRECTION_TO)
        g.addedgewithweight('3', '6', -8, _g.DIRECTION_TO)
        g.addedgewithweight('4', '1', -4, _g.DIRECTION_TO)
        g.addedgewithweight('4', '5', 3, _g.DIRECTION_TO)
        g.addedgewithweight('5', '2', 7, _g.DIRECTION_TO)
        g.addedgewithweight('6', '2', 5, _g.DIRECTION_TO)
        g.addedgewithweight('6', '3', 10, _g.DIRECTION_TO)
        print('权值矩阵为')
        W = g.getmatrixwithweight()
        print(W)
        print('迭代最短距离矩阵为')
        L = _esp.faster_all_pairs_shortest_paths(W)
        print(L)
        print('pi矩阵为')
        print(_esp.getpimatrix(g, L, W))

    def note(self):
        '''
        Summary
        ====
        Print chapter25.1 note

        Example
        ====
        ```python
        Chapter25_1().note()
        ```
        '''
        print('chapter25.1 note as follow')  
        print('第25章 每对顶点间的最短路径')
        print('在本章中，讨论找出图中每对顶点最短路径的问题')
        print('例如，对一张公路图，需要制表说明每对城市间的距离，就可能出现这种问题')
        print('给定一加权有向图G=(V,E),其加权函数w：E->R为边到实数权值的映射',
            '对于每对顶点u,v∈V，希望找出从u到v的一条最短(最小权)路径,',
            '其中路径的权值是指其组成边的权值之和')
        print('通常希望以表格形式输出结果:第u行第v列的元素应是从u到v的最短路径的权值')
        print('可以把单源最短路径算法运行|V|次来解决每对顶点间最短路径问题,每一次运行时,',
            '轮流把每个顶点作为源点。如果所有边的权值是非负的,可以采用Dijkstra算法')
        print('如果采用线性数组来实现最小优先队列,算法的运行时间为O(V^3+VE)=O(V^3)')
        print('如果是稀疏图,采用二叉最小堆来实现最小优先队列,就可以把算法的运行时间改进为O(VElgV)')
        print('或者采用斐波那契堆来实现最小优先队列,其算法运行时间为O(V^2lgV+VE)')
        print('如果允许有负权值的边,就不能采用Dijkstra算法.必须对每个顶点运行一次速度较慢的Bellman-Ford算法',
            '它的运行时间为O(V^2E),而在稠密图上的运行时间为O(V^4)')
        print('本章中每对顶点间最短路径算法的输出是一个n*n的矩阵D=(dij)',
            '其中元素dij是从i到j的最短路径的权值。就是说，如果用d(i,j)表示从顶点i到顶点j的最短路径的权值',
            '则在算法终止时dij=d(i,j)')
        print('为了求解对输入邻接矩阵的每对顶点间最短路径问题,不仅要算出最短路径的权值,而且要计算出一个前驱矩阵∏',
            '其中若i=j或从i到j没有通路,则pi(i,j)为None,否则pi(i,j)表示从i出发的某条最短路径上j的前驱顶点')
        print('25.1节介绍一个基于矩阵乘法的动态规划算法,求解每对顶点间的最短路径问题',
            '由于采用了重复平方的技术,算法的运行时间为Θ(V^3lgV)')
        print('25.2节给出另一种动态规划算法,即Floyd-Warshall算法,该算法的运行时间为Θ(V^3)')
        print('25.2节还讨论求有向图传递闭包的问题,这一问题与每对顶点间最短路径有关系')
        print('25.3节介绍Johnson算法,Johnson算法采用图的邻接表表示法',
            '该算法求解每对顶点间最短路径问题所需的时间为O(V^2lgV+VE)',
            '对大型稀疏图来说这是一个很好的算法')
        print('25.1 最短路径与矩阵乘法')
        print('动态规划算法,用来解决有向图G=(V,E)上每对顶点间的最短路径问题')
        print('动态规划的每一次主循环都将引发一个与矩阵乘法运算十分相似的操作',
            '因此算法看上去很像是重复的矩阵乘法','开始先找到一种运行时间为Θ(V^4)的算法',
            '来解决每对顶点间的最短路径问题,然后改进这一算法,使其运行时间达到Θ(V^3lgV)')
        print('动态规划算法的几个步骤')
        print('1) 描述一个最优解的结构')
        print('2) 递归定义一个最优解的值')
        print('3) 按自底向上的方式计算一个最优解的值')
        print('最短路径最优解的结构')
        print('  对于图G=(V,E)上每对顶点间的最短路径问题,',
            '已经在引理24.1中证明了最短路径的所有子路径也是最短路径')
        print('假设图以邻接矩阵W=(wij)来表示,考察从顶点i到顶点j的一条最短路径p,假设p至多包含m条边',
            '假设图中不存在权值为负的回路,则m必是有限值.如果i=j,则路径p权值为0而且没有边')
        print('若顶点i和顶点j是不同顶点,则把路径p分解为i~k->j，其中路径p\'至多包含m-1条边')
        print('每对顶点间最短路径问题的一个递归解')
        print('  设lij(m)是从顶点i到顶点j的至多包含m条边的任何路径的权值最小值.',
            '当m=0时,从i到j存在一条不包含边的最短路径当且仅当i=j')
        print('  对m>=1,先计算lij(m-1),以及从i到j的至多包含m条边的路径的最小权值,',
            '后者是通过计算j的所有可能前趋k而得到的,然后取二者中的最小值作为lij(m),因此递归定义')
        print('  lij(m)=min(lij(m-1),min{lik(m-1)+wkj})=min{lik(m-1)+wkj}')
        print('  后一等式成立是因为对所有j,wij=0')
        print('自底向上计算最短路径的权值')
        print('  把矩阵W=(wij)作为输入,来计算一组矩阵L(1),L(2),...,L(n-1)',
            '其中对m=1,2,...,n-1,有L(m)=(lij(m)).最后矩阵L(n-1)包含实际的最短路径权值',
            '注意：对所有的顶点i,j∈V，lij(1)=wij,因此L(1)=W')
        print('算法的输入是给定矩阵L(m-1)和W,返回矩阵L(m),就是把已经计算出来的最短路径延长一条边')
        print('改进算法的运行时间')
        print('  目标并不是计算出全部的L(m)矩阵，所感兴趣的是仅仅是倒数第二个迭代矩阵L(n-1)',
            '如同传统的矩阵乘法满足结合律,EXTEND-SHORTEST-PATHS定义的矩阵乘法也一样')
        print('  通过两两集合矩阵序列,只需计算[lg(n-1)]个矩阵乘积就能计算出L(n-1)')
        print('  因为[lg(n-1)]个矩阵乘积中的每一个都需要Θ(n^3)时间',
            '因此FAST-ALL-PAIRS-SHORTEST-PATHS的运行时间Θ(n^3lgn)')
        print('  算法中的代码是紧凑的,不包含复杂的数据结构,因此隐含于Θ记号中的常数是很小的')
        _esp.test_show_all_pairs_shortest_paths()
        print('练习25.1-1 代码如下')
        self.solve_25_1_1()
        print('练习25.1-2 对所有的1<=i<=n,要求wii=0，因为结点对自身的最短路径始终为0')
        print('练习25.1-3 最短路径算法中使用的矩阵L(0)对应于常规矩阵乘法中的单位矩阵？')
        print('练习25.1-4 EXTEND-SHORTEST-PATHS所定义的矩阵乘法满足结合律')
        print('练习25.1-5 可以把单源最短路径问题表述为矩阵和向量的乘积.',
            '描述对该乘积的计算是如何与类似Bellman-Ford这样的算法相一致')
        print('练习25.1-6 希望在本节的算法中的出最短路径上的顶点。说明如何在O(n^3)时间内,',
            '根据已经完成的最短路径权值的矩阵L计算出前趋矩阵∏ 略')
        print('练习25.1-7 可以用于计算最短路径的权值相同的时间,计算出最短路径上的顶点')
        print('练习25.1-8 FASTER-ALL-PAIRS-SHORTEST-PATHS过程需要我们保存[lg(n-1)]个矩阵,',
            '每个矩阵包含n^2个元素,总的空间需求为Θ(n^2lgn),修改这个过程，',
            '使其仅使用两个n*n矩阵，需要空间为Θ(n^2)')
        print('练习25.1-9 修改FASTER-PARIS-SHORTEST-PATHS,使其能检测出图中是否存在权值为负的回路')
        print('练习25.1-10 写出一个有效的算法来计算图中最短的负权值回路的长度(即所包含的边数)')
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

class Chapter25_2:
    '''
    chpater25.2 note and function
    '''
    def solve_25_2_1(self):
        '''
        练习25.2-1
        '''
        g = _g.Graph()
        vertexs = ['1', '2', '3', '4']
        g.addvertex(vertexs)
        g.addedge('4', '1', _g.DIRECTION_TO)
        g.addedge('4', '3', _g.DIRECTION_TO)
        g.addedge('2', '4', _g.DIRECTION_TO)
        g.addedge('2', '3', _g.DIRECTION_TO)
        g.addedge('3', '2', _g.DIRECTION_TO)
        mat = g.getmatrixwithweight()
        print('带权邻接矩阵')
        print(mat)
        D_last = mat
        for i in range(g.vertex_num):
            print('第%d次迭代' % i)
            D = _esp.floyd_warshall_step(D_last, i)
            print(D)
            D_last = D

    def solve_25_2_6(self):
        '''
        练习25.2-6
        '''
        g = _g.Graph()
        vertexs = ['1', '2', '3', '4', '5']
        g.addvertex(vertexs)
        g.addedgewithdir('1', '2', 2)
        g.addedgewithdir('2', '3', 5)
        g.addedgewithdir('3', '4', 3)
        g.addedgewithdir('3', '5', -4)
        g.addedgewithdir('5', '2', -2)
        mat = g.getmatrixwithweight()
        print('带权邻接矩阵')
        print(mat)
        pi = g.getpimatrix()
        D, P = _esp.floyd_warshall(mat, pi)
        print('路径矩阵')
        print(D)
        print('前趋矩阵')
        print(P)

    def note(self):
        '''
        Summary
        ====
        Print chapter25.2 note

        Example
        ====
        ```python
        Chapter25_2().note()
        ```
        '''
        print('chapter25.2 note as follow')  
        print('Floyd-Warshall算法')
        print('采取另一种动态规划方案,解决在一个有向图G=(V,E)上每对顶点间的最短路径问题')
        print('Floyd-Warshall算法,其运行时间为Θ(V^3),允许存在权值为负的边,假设不存在权值为负的回路')
        print('最短路径结构')
        print('  在Floyd-Warshall算法中,利用最短路径结构中的另一个特征',
            '不同于基于矩阵乘法的每对顶点算法中所用到的特征')
        print('  该算法考虑最短路径上的中间顶点,其中简单路径p=<v1,v2,...,vl>',
            '上的中间顶点是除v1和vl以外p上的任何一个顶点,任何属于集合{v2,v3,...,vl-1}的顶点')
        print('  Floyd—Warshall算法主要基于以下观察.设G的顶点为V={1,2,...,n},对某个k考虑顶点的一个子集{1,2,...,k}',
            '对任意一对顶点i,j∈V，考察从i到j且中间顶点皆属于集合{1,2,...,k}的所有路径')
        print('  设p是其中一条最小权值路径(路径p是简单的),Floyd-Warshall算法利用了路径p与i到j之间的最短路径',
            '(所有中间顶点都属于集合{1,2,...,k-1})之间的联系.这一联系依赖于k是否是路径p上的一个中间顶点')
        print(' 分成两种情况：如果k不是路径p的中间顶点，则p的所有中间顶点皆在集合{1,2,...,k-1}中',
            '如果k是路径p的中间顶点,那么可将p分解为i~k~j,由引理24.1可知,p1是从i到k的一条最短路径',
            '且其所有中间顶点均属于集合{1,2,...,k-1}')
        print('解决每对顶点间最短路径问题的一个递归解')
        print('  令dij(k)为从顶点i到顶点j、且满足所有中间顶点皆属于集合{1,2,...,k}的一条最短路径的权值',
            '当k=0,从顶点i到顶点j的路径中,没有编号大于0的中间顶点')
        print('递归式')
        print('  dij(k)=wij e.g. k = 0;  dij(k)=min(dij(k-1),dik(k-1),dkj(k-1)) e.g. k >= 1;')
        print('  因为对于任意路径，所有的中间顶点都在集合{1,2,...,n}内,矩阵D(n)=dij(n)给出了最终解答',
            '对所有的i,j∈V,有dij(n)=d(i,j)')
        print('自底向上计算最短路径的权值')
        _esp.test_floyd_warshall()
        print('构造一条最短路径')
        print('  在Floyd—Warshall算法中存在大量不同的方法来建立最短路径')
        print('  一种途径是计算最短路径权值的矩阵D,然后根据矩阵D构造前趋矩阵pi。这一方法可以在O(n^3)时间内实现',
            '给定前趋矩阵pi,可以使用过程PRINT-ALL-PAIRS-SHORTEST-PATH来输出一条给定最短路径上的顶点')
        print('有向图的传递闭包')
        print('  已知一有向图G=(V,E),顶点集合V={1,2,..,n},希望确定对所有顶点对i,j∈V',
            '图G中是否都存在一条从i到j的路径.G的传递闭包定义为图G*=(V,E*)',
            '其中E*={(i,j):图G中存在一条从i到j的路径}')
        print('  在Θ(n^3)时间内计算出图的传递闭包的一种方法为对E中每条边赋以权值1,然后运行Floyd-Warshall算法',
            '如果从顶点i到顶点j存在一条路径,则dij<n。否则,有dij=∞')
        print('  另外还有一种类似的方法,可以在Θ(n^3)时间内计算出图G的传递闭包,在实际中可以节省时空需求',
            '该方法要求把Floyd-Warshall算法中的min和+算术运算操作,用相应的逻辑运算∨(逻辑OR)和∧(逻辑AND)来代替)',
            '对i,j,k=1,2,...,n,如果图G中从顶点i到顶点j存在一条通路,且其所有中间顶点均属于集合{1,2,...,k}',
            '则定义tij(k)为1,否则tij(k)为0.我们把边(i,j)加入E*中当且仅当tij(n)=1',
            '通过这种方法构造传递闭包G*=(V,E*)')
        print('tij(k)的递归定义为',
            'tij(0) = 0 e.g. 如果i!=j和(i,j)∉E',
            'tij(0) = 1 e.g. 如果i=j或(i,j)∈E',
            'k>=1, tij(k) = tij(k-1) or (tik(k-1) and tkj(k-1))')
        _esp.test_transitive_closure()
        print('练习25.2-1 代码如下')
        self.solve_25_2_1()
        print('练习25.2-2 略')
        print('练习25.2-3 证明对所有的i∈V,前趋子图Gpi,i是以i为根的一颗最短路径树')
        print('练习25.2-4 Floyd-Washall的空间复杂度可以从Θ(n^3)优化到Θ(n^2)，完成')
        print('练习25.2-5 正确')
        print('练习25.2-6 可以利用Floyd-Warshall算法的输出来检测是否存在负的回路',
            '从最终前趋矩阵看出每个路径都相同，且为负权回路')
        self.solve_25_2_6()
        print('练习25.2-7 略')
        print('练习25.2-8 写出一个运行时间为O(VE)的算法,计算有向图G=(V,E)的传递闭包')
        print('练习25.2-9 假定一个有向无环图的传递闭包可以在f(|V|,|E|)时间内计算,其中f是|V|和|E|的单调递增函数',
            '证明：计算一般有向图G=(V,E)的传递闭包G*=(V,E*)的时间为f(|V|,|E|)+O(V+E*)')
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

class Chapter25_3:
    '''
    chpater25.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter25.3 note

        Example
        ====
        ```python
        Chapter25_3().note()
        ```
        '''
        print('chapter25.3 note as follow')  
        print('25.3 稀疏图上的Johnson算法')
        print('Johnson算法可在O(V^2lgV+VE)时间内,求出每对顶点间的最短路径.对于稀疏图',
            '该算法在渐进意义上要好于矩阵的重复平方或Floyd-Warshall算法',
            '算法执行后,返回一个关于每对顶点间最短路径的权值的矩阵',
            '或者报告输入图中存在一个负权值的回路')
        print('Johnson算法把第24章描述的Dijkstra算法和Bellman-Ford算法作为其子程序')
        print('Johnson算法运用了重赋权技术，其执行方式如下,如果图G=(V,E)中所有边的权w均为非负,',
            '则把每对顶点依次作为源点来执行Dijkstra算法,就可以找出每对顶点间的最短路径;',
            '利用斐波那契堆最小优先队列,则算法的运行时间为O(V^2lgV+VE)')
        print('如果图G=(V,E)中含有负权边单不含有负权的回路,就只计算一个新的负权边的集合,而这可以采用相同的方法',
            '这个新的边权值w的集合必须满足两个重要性质:')
        print(' 1) 对所有顶点对u,v∈V,路径p是利用加权函数w从u到v的一条最短路径,',
            '当且仅当p也是利用加权函数w从u到v的一条最短路径')
        print(' 2) 对于所有的边(u,v)，新的权w(u,v)是非负的')
        print('为确定新的加权函数w而对G进行的预处理可在O(VE)时间内完成')
        print('通过重赋权值保持最短路径')
        print('  引理25.1(重赋权值不会改变最短路径) 已知带权有向图G=(V,E),加权函数为w:',
            'E->R,设h:V->R是将顶点映射到实数的任意函数.对每条边(u,v)∈E,定义')
        print('  w`(u,v) = w(u,v) + h(u) - h(v)')
        print('  令p=<v0,v1,...,vk>为从顶点v0到顶点vk的任意一条路径.则p是利用加权函数w从v0到vk的一条最短路径',
            '当且仅当p也是利用加权函数w`的一条最短路径,亦即w(p)=d(v0,vk)当且仅当w`(p)=d(v0,vk)',
            '另外,使用加权函数w时,G中存在一条负权回路,当且仅当使用加权函数w`时,G中存在一条负权的回路')
        print('通过重赋权产生非负的权')
        print('  希望对于所有边(u,v)∈V,w`(u,v)的值非负.给定一带权有向图G=(V,E),加权函数为w：E->R',
            '据此构造一个新图G`=(V`,E`),其中对某个新顶点s∉V,V`=V∪{s},E`=E∪{(s,v),v∈V}')
        print('  扩展加权函数w,使得对所有的v∈V,有w(s,v)=0.注意因为不存在进入顶点s的边',
            '所以除了以s作为源点的路径,G`中不存在包含s的其他最短路径',
            '再者,G`不包含负权回路,当且仅当G不包含负权回路')
        print('计算每对顶点间的最短路径')
        print('  在计算每对顶点间最短路径的Johnson算法中,',
            '把Bellman-Ford算法和Dijkstra算法作为其子程序')
        print('  算法假设图的边用邻接表形式存储.算法返回通常|V|*|V|矩阵D=dij',
            '其中dij=d(i,j),或者报告输入图中存在一负权的回路')
        print('如果采用斐波那契堆来实现Dijkstra算法中的最小优先队列,则Johnson算法的云个性时间为O(V^2lgV+VE)',
            '更简单的二叉堆实现,则可以得到O(VElgV)的运行时间')
        print('对于稀疏图来说,Johnson算法在渐进意义上仍然比Floyd-Warshall算法快')
        print('练习25.3-1 代码如下')
        _esp.test_johnson()
        print('练习25.3-2 把新顶点s加入到V中得到V`是因为G`不存在包含s的其他最短路径')
        print('练习25.3-3 假定对所有边(u,v)∈E,有w(u,v)>=0,那么加权函数w和w`有什么关系')
        print('练习25.3-4 GreenStreet重赋权方法是错误的')
        print('练习25.3-5 证明：在有向图G上用加权函数w运行Johnson算法,如果G包含0权的回路c,则对c中的每个边(u,v),w`(u,v)=0')
        print('练习25.3-6 如果G是强连通的(每个顶点都可以和其他每个顶点相连接),Johnson算法可以不加入新的源顶点，并且答案正确')
        print('思考题25-1 动态图的传递闭包')
        print('  假设对有向图G=(V,E),插入边到E中时希望保持传递闭包的正确性',
            '即在插入每条边后,希望对已插入边的传递闭包进行更新',
            '假设图G开始不含任何边,并且传递闭包用布尔矩阵来表示')
        print('  a) 当插入一条新边到图G=(V,E)时,如何能在O(V^2)时间内对其传递闭包G*=(V,E*)进行更新')
        print('  b) 举出一图G和边e的例子,当e被插入到G中后,需要Ω(V^2)的运行时间对G的传递闭包进行更新')
        print('  c) 描述一个有效的算法,使得在图中插入一条边时,算法能对图的传递闭包进行更新')
        print('思考题25-2 ε稠密图中的最短路径')
        print('  在图G=(V,E)中,如果对某常数0<ε<=1,有|E|=Θ(V^(1+ε)),则说G是ε稠密的')
        print('  通过在ε稠密图上的最短路径算法中应用d叉最小堆,',
            '能使算法的运行时间相当于基于斐波那契堆的算法的运行时间')
        # python src/chapter25/chapter25note.py
        # python3 src/chapter25/chapter25note.py

chapter25_1 = Chapter25_1()
chapter25_2 = Chapter25_2()
chapter25_3 = Chapter25_3()

def printchapter25note():
    '''
    print chapter25 note.
    '''
    print('Run main : single chapter twenty-five!')  
    chapter25_1.note()
    chapter25_2.note()
    chapter25_3.note()

# python src/chapter25/chapter25note.py
# python3 src/chapter25/chapter25note.py
if __name__ == '__main__':  
    printchapter25note()
else:
    pass

```

```py

import graph as _g
import math as _math
from copy import deepcopy as _deepcopy

class _ShortestPath:
    '''
    单源最短路径算法集合
    '''
    def __init__(self, *args, **kwords):
        pass

    def initialize_single_source(self, g : _g.Graph, s : _g.Vertex):
        '''
        最短路径估计和前趋进行初始化 时间复杂度Θ(V)
        '''
        for v in g.veterxs:
            v.d = _math.inf
            v.pi = None
        s.d = 0

    def relax(self, u : _g.Vertex, v : _g.Vertex, weight):
        '''
        一步松弛操作
        '''
        if v.d > u.d + weight:
            v.d = u.d + weight
            v.pi = u
    
    def bellman_ford(self, g : _g.Graph, s : _g.Vertex):
        '''
        Bellmax-Ford算法能在一般的情况下(存在负边权的情况)下,解决单源最短路径问题

        时间复杂度 O(VE)

        Args
        ===
        `g` : 图G=(V,E)

        `s` : 源顶点

        Return
        ===
        `exist` : bool 返回一个布尔值,表明图中是否存在着一个从源点可达的权为负的回路
        若存在这样的回路的话,算法说明该问题无解;若不存在这样的回路,算法将产生最短路径以及权值

        `weight` : 权值

        '''
        weight = 0
        if type(s) is not _g.Vertex:
            s = g.veterxs_atkey(s)
        self.initialize_single_source(g, s)
        n = g.vertex_num
        for i in range(n - 1):
            for edge in g.edges:
                u, v = edge.vertex1, edge.vertex2
                u = g.veterxs_atkey(u)
                v = g.veterxs_atkey(v)
                self.relax(u, v, edge.weight)
        for edge in g.edges:
            u, v = edge.vertex1, edge.vertex2
            u = g.veterxs_atkey(u)
            v = g.veterxs_atkey(v)
            if v.d > u.d + edge.weight:
                return False, weight
            weight += edge.weight
        return True, weight
    
    def dag_shortest_path(self, g : _g.Graph, s : _g.Vertex):
        '''
        按顶点的拓扑序列对某加权dag图(有向无回路图)G=(V,E)的边进行松弛后
        就可以在Θ(V+E)时间内计算出单源最短路径.

        Args
        ===
        `g` : 有向无回路图G=(V,E) 

        `s` : 源顶点

        '''
        sort_list = _g.topological_sort(g)
        self.initialize_single_source(g, s)
        for u in sort_list:
            u = g.veterxs_atkey(u)
            adj = g.getvertexadj(u)
            for v in adj:
                edge = g.getedge(u, v)
                self.relax(u, v, edge.weight)
            
    def dijstra(self, g : _g.Graph, s : _g.Vertex):
        '''
        单源最短路径Dijstra算法
        '''
        self.initialize_single_source(g, s)
        S = []
        Q = _deepcopy(g.veterxs)
        while len(Q) != 0:
            Q.sort(reverse=True)
            u = Q.pop()
            S += [u]
            adj = g.getvertexadj(u)
            if adj is not None:
                for v in adj:
                    edge = g.getedge(u, v)
                    self.relax(u, v, edge.weight)

__shortest_path_instance = _ShortestPath()
bellman_ford = __shortest_path_instance.bellman_ford
dag_shortest_path = __shortest_path_instance.dag_shortest_path
dijstra = __shortest_path_instance.dijstra

def test_bellman_ford():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('s'), _g.Vertex('t'), _g.Vertex(
        'x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('s', 't', 6, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'y', 7, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 5, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 8, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', -4, _g.DIRECTION_TO)
    g.addedgewithweight('x', 't', -2, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'x', -3, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', 9, _g.DIRECTION_TO)
    g.addedgewithweight('z', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('z', 's', 2, _g.DIRECTION_TO)
    print(bellman_ford(g, vertexs[0]))
    del g

def test_dag_shortest_path():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dag_shortest_path(g, vertexs[0])
    del g

def test_dijstra():
    g = _g.Graph()
    g.clear()
    vertexs = [_g.Vertex('r'), _g.Vertex('s'), _g.Vertex('t'),
        _g.Vertex('x'), _g.Vertex('y'), _g.Vertex('z')]
    g.veterxs = vertexs
    g.addedgewithweight('r', 's', 5, _g.DIRECTION_TO)
    g.addedgewithweight('s', 't', 2, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'x', 7, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'y', -1, _g.DIRECTION_TO)
    g.addedgewithweight('y', 'z', -2, _g.DIRECTION_TO)
    g.addedgewithweight('r', 't', 3, _g.DIRECTION_TO)
    g.addedgewithweight('s', 'x', 6, _g.DIRECTION_TO)
    g.addedgewithweight('x', 'z', 1, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'y', 4, _g.DIRECTION_TO)
    g.addedgewithweight('t', 'z', 2, _g.DIRECTION_TO)
    g.reset_vertex_para()
    dijstra(g, vertexs[0])
    del g

def test():
    '''
    测试函数
    '''
    test_bellman_ford()
    test_dag_shortest_path()
    test_dijstra()

if __name__ == '__main__':
    test()
else:
    pass

```

```py

import math as _math
from copy import deepcopy as _deepcopy
import numpy as _np

import graph as _g
import shortestpath as _sp

class _ExtendShortestPath:
    '''
    扩展最短路径对算法集合类
    '''
    def __init__(self, *args, **kwords):
        '''
        扩展最短路径对算法集合类
        '''
        pass

    def extend_shortest_paths(self, L, W):
        '''
        最短路径对矩阵`L`单步迭代过程
        '''
        n = _np.shape(L)[0] # rows of L
        L_return = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                L_return[i][j] = _math.inf
                for k in range(n):
                    L_return[i][j] = min(L_return[i][j], L[i][k] + W[k][j])
        return L_return

    def show_all_pairs_shortest_paths(self, W):
        '''
        根据图`g`的权值矩阵`W`求所有最短路径对矩阵
        '''
        n = _np.shape(W)[0] # rows of W
        L = list(range(n))
        L[0] = W
        for m in range(1, n - 1):
            L[m] = self.extend_shortest_paths(L[m - 1], W)
        return L[n - 2]

    def faster_all_pairs_shortest_paths(self, W):
        '''
        根据图`g`的权值矩阵`W`求所有最短路径对矩阵
        '''
        n = _np.shape(W)[0] # rows of W
        L_last = W
        L_now = []
        m = 1
        while m < n - 1:
            L_now = self.extend_shortest_paths(L_last, L_last)
            m = 2 * m
            L_last = L_now
        return L_now

    def getpimatrix(self, g, L, W):
        '''
        获得前趋矩阵`∏`
        '''
        n = _np.shape(W)[0] # rows of W
        pi = _np.zeros((n, n), dtype=_np.str)
        index = 0
        for i in range(n):
            for j in range(n):
                pi[i][j] = '∞'
                if i == j:
                    pi[i][j] = g.veterxs[i].key
                else:
                    if L[i][j] == _math.inf:
                        pi[i][j] = '∞'
                        continue
                    if L[i][j] == W[i][j]:
                        pi[i][j] = g.veterxs[j].key
                        continue
                    for k in range(n):
                        if k != i and k !=j and L[i][j] == L[i][k] + L[k][j]:
                            pi[i][j] = g.veterxs[k].key
        return pi

    def floyd_warshall_step(self, D_last, k):
        '''
        单步`Floyd-Warshall`算法
        '''
        n = _np.shape(D_last)[0] # rows of W
        D = _deepcopy(D_last)
        for i in range(n):
            for j in range(n):
                D[i][j] = min(D_last[i][j], D_last[i][k] + D_last[k][j])
        return D

    def floyd_warshall(self, W, pi):
        '''
        根据图`g`的权值矩阵`W`求最短路径对矩阵的`Floyd-Warshall`算法
        '''
        n = _np.shape(W)[0] # rows of W
        D = W
        D_last = W
        P = pi
        P_last = pi
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D_last[i][j] > D_last[i][k] + D_last[k][j]:
                        D[i][j] = D_last[i][k] + D_last[k][j]
                        P[i][j] = P_last[k][j]
                    else:
                        D[i][j] = D_last[i][j]
                        P[i][j] = P_last[i][j]
            D_last = D
            P_last = P
        return D, P

    def transitive_closure(self, g : _g.Graph):
        '''
        有向图`g`的传递闭包
        '''
        n = g.vertex_num
        t = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                edge = g.getedge(g.veterxs[i], g.veterxs[j])
                if i == j or edge in g.edges:
                    t[i][j] = 1
                else:
                    t[i][j] = 0
        t_last = _deepcopy(t)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    result = t_last[i][j] or (t_last[i][k] and t_last[k][j])
                    if result == True:
                        t[i][j] = 1
                    else:
                        t[i][j] = 0
            t_last = t
        return t

    def johnson(self, g : _g.Graph):
        '''
        根据图`g`的邻接表`adj`求最短路径对矩阵的`Johnson`算法
        '''
        new_g = _deepcopy(g)
        s = _g.Vertex('0')
        # V[G`] = V[G]∪{s}
        new_g.addvertex(s)
        # E[G`] = E[G]∪{(s,v),v∈V}
        # w(s, v) = 0 for all v∈V[G]
        for v in new_g.veterxs:
            new_g.addedgewithdir(s, v, 0)
        exist, weight = _sp.bellman_ford(new_g, s)
        if exist == False:
            print('the graph contains a negative-weight cycle')
        else:
            n = new_g.vertex_num
            D = _np.zeros((n, n))
            for v in new_g.veterxs:
                v.d = weight
            for edge in new_g.edges:
                u, v = new_g.getvertexfromedge(edge)
                edge.weight = edge.weight + u.d - v.d
            for u in new_g.veterxs:
                _sp.dijstra(new_g, u)
                uindex = new_g.getvertexindex(u)
                for v in new_g.veterxs:
                    vindex = new_g.getvertexindex(v)
                    edge = new_g.getedge(u, v)
                    if edge is not None:
                        D[uindex][vindex] = edge.weight + v.d - u.d
            return D
        
__esp_instance = _ExtendShortestPath()

extend_shortest_paths = __esp_instance.extend_shortest_paths
show_all_pairs_shortest_paths = __esp_instance.show_all_pairs_shortest_paths
faster_all_pairs_shortest_paths = __esp_instance.faster_all_pairs_shortest_paths
getpimatrix = __esp_instance.getpimatrix
floyd_warshall_step = __esp_instance.floyd_warshall_step 
floyd_warshall = __esp_instance.floyd_warshall
transitive_closure = __esp_instance.transitive_closure
johnson = __esp_instance.johnson

def test_show_all_pairs_shortest_paths():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4', '5']
    g.addvertex(vertexs)
    g.addedgewithweight('1', '2', 3, _g.DIRECTION_TO)
    g.addedgewithweight('1', '3', 8, _g.DIRECTION_TO)
    g.addedgewithweight('1', '5', -4, _g.DIRECTION_TO)
    g.addedgewithweight('2', '4', 1, _g.DIRECTION_TO)
    g.addedgewithweight('2', '5', 7, _g.DIRECTION_TO)
    g.addedgewithweight('3', '2', 4, _g.DIRECTION_TO)
    g.addedgewithweight('4', '1', 2, _g.DIRECTION_TO)
    g.addedgewithweight('4', '3', -5, _g.DIRECTION_TO)
    g.addedgewithweight('5', '4', 6, _g.DIRECTION_TO)
    W = g.getmatrixwithweight()
    print('带权值的邻接矩阵为：')
    print(W)
    print('显示所有的最短路径')
    print(show_all_pairs_shortest_paths(W))
    print('显示所有的最短路径(对数加速)')
    L = faster_all_pairs_shortest_paths(W)
    print(L)
    print('pi矩阵为')
    pi = getpimatrix(g, L, W)
    print(pi)

    print(show_all_pairs_shortest_paths([[0,5,4],[_math.inf, 0, 7], [1, _math.inf, 0]]))
    print(show_all_pairs_shortest_paths([[0,5,4],[-1, 0, 7], [1, -1, 0]]))

def test_floyd_warshall():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4', '5']
    g.addvertex(vertexs)
    g.addedgewithweight('1', '2', 3, _g.DIRECTION_TO)
    g.addedgewithweight('1', '3', 8, _g.DIRECTION_TO)
    g.addedgewithweight('1', '5', -4, _g.DIRECTION_TO)
    g.addedgewithweight('2', '4', 1, _g.DIRECTION_TO)
    g.addedgewithweight('2', '5', 7, _g.DIRECTION_TO)
    g.addedgewithweight('3', '2', 4, _g.DIRECTION_TO)
    g.addedgewithweight('4', '1', 2, _g.DIRECTION_TO)
    g.addedgewithweight('4', '3', -5, _g.DIRECTION_TO)
    g.addedgewithweight('5', '4', 6, _g.DIRECTION_TO)
    W = g.getmatrixwithweight()
    print('带权值的邻接矩阵为：')
    print(W)
    print('初始前趋矩阵')
    pi = g.getpimatrix()
    print(pi)
    print('所有的最短路径')
    D, P = floyd_warshall(W, pi)
    print(D)
    print('最终前趋路径')
    print(P)

def test_transitive_closure():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4']
    g.addvertex(vertexs)
    g.addedge('4', '1', _g.DIRECTION_TO)
    g.addedge('4', '3', _g.DIRECTION_TO)
    g.addedge('2', '4', _g.DIRECTION_TO)
    g.addedge('2', '3', _g.DIRECTION_TO)
    g.addedge('3', '2', _g.DIRECTION_TO)
    mat = g.getmatrix()
    print('邻接矩阵')
    print(mat)
    t = transitive_closure(g)
    print('传递闭包')
    print(t)

def test_johnson():
    g = _g.Graph()
    vertexs = ['1', '2', '3', '4']
    g.addvertex(vertexs)
    g.addedge('4', '1', _g.DIRECTION_TO)
    g.addedge('4', '3', _g.DIRECTION_TO)
    g.addedge('2', '4', _g.DIRECTION_TO)
    g.addedge('2', '3', _g.DIRECTION_TO)
    g.addedge('3', '2', _g.DIRECTION_TO)
    mat = g.getmatrix()
    print('邻接矩阵')
    print(mat)
    print(johnson(g))

def test():
    test_show_all_pairs_shortest_paths()
    test_floyd_warshall()
    test_transitive_closure()
    test_johnson()

if __name__ == '__main__':
    test()
else:
    pass


```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
'''

Class Chapter26_1

Class Chapter26_2

Class Chapter26_3

Class Chapter26_4

Class Chapter26_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    import flownetwork as _fn
else:
    from . import flownetwork as _fn

class Chapter26_1:
    '''
    chpater26.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.1 note

        Example
        ====
        ```python
        Chapter26_1().note()
        ```
        '''
        print('chapter26.1 note as follow')  
        print('第26章 最大流')
        print('为了求从一点到另一点的最短路径，可以把公路地图模型化为有向图')
        print('可以把一个有向图理解为一个流网络,并运用它来回答有关物流方面的问题')
        print('设想某物质从产生它的源点经过一个系统,流向消耗该物质的汇点(sink)这样一种过程')
        print('源点以固定速度产生该物质,而汇点则用同样的速度消耗该物质.',
            '从直观上看,系统中任何一点的物质的流为该物质在系统中运行的速度')
        print('物质进入某顶点的速度必须等于离开该顶点的速度,流守恒性质,',
            '当物质是电流时,流守恒与基尔霍夫电流定律等价')
        print('最大流问题是关于流网络的最简单的问题')
        print('26.1 流网络')
        print('流网络的流')
        print('  流网络G=(V,E)是一个有向图,其中每条边(u,v)∈E均有一非负容量c(u,v)>=0',
            '如果(u,v)∉E,则假定c(u,v)=0。流网络中有两个特别的顶点：源点s和汇点t',
            '为了方便起见，假定每个顶点均处于从源点到汇点的某条路径上，就是说,每个顶点v∈V,存在一条路径s->v->t',
            '因此,图G为连通图,且|E|>=|V|-1')
        print('流的定义')
        print('  设G=(V,E)是一个流网络,其容量函数为c。设s为网络的源点，t为汇点。',
            'G的流是一个实值函数f:V*V->R,且满足下列三个性质：')
        print('  (1) 容量限制：对所有u,v∈V,要求f(u,v)<=c(u,v)')
        print('  (2) 反对称性：对所有u,v∈V,要求f(u,v)=-f(v,u)')
        print('  (3) 流守恒性：对所有u∈V-{s,t},要求∑f(u,v)=0')
        print('  f(u,v)称为从顶点u到顶点v的流，可以为正，为零，也可以为负。流f的值定义为|f|=∑f(s,v)',
            '即从源点出发的总流.在最大流问题中,给出一个具有源点s和汇点t的流网络G,希望找出从s到t的最大值流')
        print('容量限制只说明从一个顶点到另一个顶点的网络流不能超过设定的容量',
            '反对称性说明从顶点u到顶点v的流是其反向流求负所得.',
            '流守恒性说明从非源点或非汇点的顶点出发的总网络流为0')
        print('定义某个顶点处的总的净流量(total net flow)为离开该顶点的总的正能量,减去进入该顶点的总的正能量')
        print('流守恒性的一种解释是这样的,即进入某个非源点非汇点顶点的正网络流，必须等于离开该顶点的正网络流',
            '这个性质(即一个顶点处的总的净流量必定为0)常常被形式化地称为\"流进等于流出\"')
        print('通常，利用抵消处理，可以将两城市间的运输用一个流来表示,该流在两个顶点之间的至多一条边上是正的')
        print('给定一个实际运输的网络流f,不能重构其准确的运输路线,如果知道f(u,v)=5,',
            '如果知道f(u,v)=5,表示有5个单位从u运输到了v,或者表示从u到v运输了8个单位,v到u运输了3个单位')
        print('本章的算法将隐式地利用抵消,假设边(u,v)有流量f(u,v).在一个算法的过程中,可能对边(u,v)上的流量增加d',
            '在数学上,这一操作为f(u,v)减d；从概念上看,可以认为这d个单位是对边(u,v)上d个单位流量的抵消')
        print('具有多个源点和多个汇点的网络')
        print('  在一个最大流问题中,可以有几个源点和几个汇点,而非仅有一个源点和一个汇点',
            '比如物流公司实际可能拥有m个工厂{s1,s2,...,sm}和n个仓库{t1,t2,...,tn}',
            '这个问题不比普通的最大流问题更难')
        print('  在具有多个源点和多个汇点的网络中,确定最大流的问题可以归约为一个普通的最大流问题',
            '通过增加一个超级源点s,并且对每个i=1,2,...,m加入有向边(s,si),其容量c(s,si)=∞',
            '同时创建一个超级汇点t,并且对每个j=1,2,...,n加入有向边(tj,t),其容量c(tj,t)=∞')
        print('  单源点s对多个源点si提供了其所需要的任意大的流.同样,单汇点t对多个汇点tj消耗其所需要的任意大的流')
        print('对流的处理')
        print('  下面来看一些函数(如f),它们以流网络中的两个顶点作为自变量',
            '在本章,将使用一种隐含求和记号,其中任何一个自变量或两个自变量可以是顶点的集合',
            '它们所表示的值是对自变量所代表元素的所有可能的情形求和')
        print('  流守恒限制可以表述为对所有u∈V-{s,t},有f(u,V)=0,',
            '同时,为方便起见,在运用隐含求和记法时,省略集合的大括号.例如,在等式f(s,V-s)=f(s,V)中',
            '项V-s是指集合V-{s}')
        print('隐含集合记号常可以简化有关流的等式.下列引理给出了有关流和隐含记号的几个恒等式')
        print('引理26.1 设G=(V,E)是一个流网络,f是G中的一个流.那么下列等式成立')
        print(' 1) 对所有X∈V,f(X,X)=0')
        print(' 2) 对所有X,Y∈V,f(X,Y)=-f(Y,X)')
        print(' 3) 对所有X,Y,Z∈V,其中X∧Y!=None,有f(X∨Y,Z)=f(X,Y)+f(Y,Z)且f(Z,X∨Y)=f(Z,X)+f(Z,Y)')
        print('作为应用隐含求和记法的一个例子,可以证明一个流的值为进入汇点的全部网络流,即|f|=f(V,t)')
        print('根据流守恒特性,除了源点和汇点以外,对所有顶点来说,进入顶点的总的正流量等于离开该顶点的总的正能量',
            '根据定义,源点顶点总的净流量大于0；亦即，对源点顶点来说，离开它的正流要比进入它的正流更多',
            '对称地，汇点顶点是唯一一个其总的净流量小于0的顶点;亦即,进入它的正流要比离开它的正流更多')
        print('练习26.1-1 利用流的定义，证明如果(u,v)∉E,且(v,u)∉E,有f(u,v)=f(v,u)=0')
        print('练习26.1-2 证明：对于任意非源点非汇点的顶点v,进入v的总正向流必定等于离开v的总正向流')
        print('练习26.1-3 证明在具有多个源点和多个汇点的流网络中,任意流均对应于通过增加一个超级源点',
            '和超级汇点所得到的具有相同值的一个单源点单汇点流')
        print('练习26.1-4 证明引理26.1')
        print('练习26.1-5 在所示的流网络G=(V,E)和流f,找出两个子集合X,Y∈V,且满足f(X,Y)=-f(V-X,Y)',
            '再找出两个子集合X,Y∈V,且满足f(X,Y)!=-f(V-X,Y)')
        print('练习26.1-6 给定流网络G=(V,E),设f1和f2为V*V到R上的函数.流的和f1+f2是从V*V到R上的函数',
            '定义如下：对所有u,v∈V (f1+f2)(u,v)=f1(u,v)+f2(u,v)',
            '如果f1和f2为G的流,则f1+f2必满足流的三条性质中的哪一条')
        print('练习26.1-7 设f为网络中的一个流,a为实数。标量流之积是一个从V*V到R上的函数,定义为(af)(u,v)=a*f(u,v)',
            '证明网络中的流形成一个凸集','即证明如果f1和f2是流,则对所有0<=a<=1,af1+(1-a)f2也是流')
        print('练习26.1-8 将最大流问题表述为一个线性规划问题')
        print('练习26.1-9 略')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_2:
    '''
    chpater26.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.2 note

        Example
        ====
        ```python
        Chapter26_2().note()
        ```
        '''
        print('chapter26.2 note as follow')  
        print('Ford-Fulkerson方法')
        print('解决最大流问题的Ford-Fulkerson方法,包含具有不同运行时间的几种实现')
        print('Ford-Fulkerson方法依赖于三种重要思想')
        print(' 残留网络,增广路径,割')
        print('这些思想是最大流最小割定理的精髓,该定理用流网络的割来描述最大流的值')
        print('Ford-Fulkerson方法是一种迭代方法.开始时,对所有u,v∈V,有f(u,v)=0,即初始状态时流的值为0',
            '在每次迭代中,可通过寻找一条\"增广路径\来增加流的值"。增广路径可以看作是从源点s到汇点t之间的一条路径',
            '沿该路径可以压入更多的流,从而增加流的值.反复进行这一过程,直至增广路径都被找出为止',
            '最大流最小割定理将说明在算法终止时,这一过程可产生出最大流')
        print('残留网络')
        print('  直观上,给定流网络和一个流,其残留网络由可以容纳更多网络流的边所组成',
            '更形式地,假定有一个网络G=(V,E),其源点为s,汇点到t.设f为G中的一个流,并考察一对顶点u,v∈V',
            '在不超过容量c(u,v)的条件下,从u到v之间可以压入的额外网络流量,就是(u,v)的残留容量(residual capacity),由下式定义:',
            'cf(u,v)=c(u,v)-f(u,v)')
        print('  例如,如果c(u,v)=16且f(u,v)=11,则在不超过边(u,v)的容量限制的条件下,可以再传输cf(u,v)=5个单位的流来增加f(u,v)',
            '当网络流f(u,v)为负值时,残留容量cf(u,v)大于容量c(u,v)')
        print('  例如,如果c(u,v)=16且f(u,v)-4,残留容量cf(u,v)为20')
        print('  解释：从v到u存在着4个单位的网络流,可以通过从u到v压入4个单位的网络来抵消它',
            '然后,在不超过边(u,v)的容量限制的条件下,还可以从u到v压入另外16个单位的网络流',
            '因此,从开始的网络流f(u,v)-4,共压入了额外的20个单位的网络流,并不会超过容量限制')
        print('  在残留网络中,每条边(或称为残留边)能够容纳一个严格为正的网络流')
        print('Ef中的边既可以是E中的边,也可以是它们的反向边',
            '如果边(u,v)∈E有f(u,v)<c(u,v),那么cf(u,v)=c(u,v)-f(u,v)>0且(u,v)属于Ef')
        print('只有当两条边(u,v)和(v,u)中,至少有一条边出现于初始网络中时,边(u,v)才能够出现在残留网络中,所以有如下限制条件:',
            '|Ef|<=2|E|.残留网络Gf本身也是一个流网络,其容量由cf给出.下列引理说明残留网络中的流与初始网络中的流有何关系')
        print('引理26.2 设G=(V,E)是源点为s,汇点为t的一个流网络,且f为G中的一个流',
            '设Gf是由f导出的G的残留网络,且f’为Gf中的一个流,其值|f+f`|=|f|+|f`|')
        print('增广路径')
        print('  已知一个流网络G=(V+E)和流f,增广路径p为残留网络Gf中从s到t的一条简单路径',
            '根据残留网络的定义,在不违反边的容量限制条件下,增广路径上的每条边(u,v)可以容纳从u到v的某额外正网络流')
        print('引理26.3 设G=(V,E)是一个网络流,f是G的一个流,并设p是Gf中的一条增广路径.',
            '用下式定义一个函数：fp：V*V->R')
        print('fp(u,v)=cf(p);fp(u,v)=-cf(p);fp(u,v)=0')
        print('则fp是Gf上的一个流,其值为|fp|=cf(p)>0')
        print('推论26.4 设G=(V,E)是一个流网络,f是G的一个流,p是Gf中的一条增广路径')
        print('流网络的割')
        print('  Ford-Fulkerson方法沿增广路径反复增加流,直至找出最大流为止.',
            '要证明的最大流最小割定理：一个流是最大流,当且仅当它的残留网络不包含增广路径')
        print('流网络G=(V,E)的割(S,T)将V划分成S和T=V-S两部分,使得s∈S,t∈T')
        print('一个网络的最小割也就是网络中所有割中具有最小容量的割')
        print('引理26.5 设f是源点s,汇点为t的流网络G中的一个流.并且(S,T)是G的一个割',
            '则通过割(S,T)的净流f(S,T)=|f|')
        print('推论26.6 对一个流网络G中任意流f来说,其值的上界为G的任意割的容量')
        print('定理26.7(最大流最小割定理) 如果f是具有源点s和汇点t的流网络G=(V,E)中的一个流,则下列条件是等价的:',
            '1) f是G的一个最大流')
        print('2) 残留网络Gf不包含增广路径')
        print('3) 对G的某个割(S,T),有|f|=c(S,T)')
        print('基本的Ford-Fulkerson算法')
        print('  在Ford-Fulkerson方法的每次迭代中,找出任意增广路径p,并把沿p每条边的流f加上其残留容量cf(p)',
            '在Ford-Fulkerson方法的以下实现中,',
            '通过更新有边相连的每对顶点u,v之间网络流f[u,v],来计算出图G=(V,E)中的最大流')
        print('如果u和v之间在任意方向没有边相连,则隐含地假设f[u,v]=0',
            '假定已经在图中给出,且如果(u,v)∉E,有c(u,v)=0.残留容量cf(u,v)',
            '代码中的符号cf(p)实际上只是存储路径p的残留容量的一个临时变量')
        print('Ford-Fulkerson算法的分析')
        print('  Ford-Fulkerson算法过程的运行时间取决于如何确定第4行中的增广路径,',
            '如果选择不好,算法甚至可能不会终止;流的值随着求和运算将不断增加,但它甚至不会收敛到流的最大值')
        print('  如果采用广度优先搜索来选择增广路径,算法的运行时间为多项式时间复杂度')
        print('  但是,在证明这一点之前,先任意选择增广路径、且所有容量均为整数,取得一个简单的界')
        print('在实际中碰到的大多数最大流的问题中其容量经常为整数,如果容量为有理数,则经过适当的比例转换,都可以使它们变为整数')
        print('在这一假设下,Ford-Fulkerson的一种简易实现的运行时间为O(E|f*|)')
        print('Edmonds-Karp算法')
        print('  如果在Ford-Fulkerson算法使用广度优先搜索来实现对增广路径p的计算',
            '即如果增广路径是残留网格中从s到t的最短路径(其中每条边为单位距离或权)',
            '则能够改进FORD-FULKERSON的界,这种方法称为Edmonds-Karp算法,运行时间为O(VE^2)')
        print('引理26.8 如果对具有源点s和汇点t的流网络G=(V,E)运行Edmonds-Karp算法,则对所有顶点v∈V-{s,t}',
            '残留网络Gf中的最短路径长度df(u,v)随着每个流的增加而单调递增')
        print('定理26.9 如果对具有源点s和汇点t的一个流网络G=(V,E)运行Edmonds-Karp算法,对流进行增加的全部次数为O(VE)')
        print('  在一个残留网络Gf中,如果对其增广路径p的残留容量是边(u,v)的残留容量,即,如果cf(p)=cf(u,v),则说边(u,v)对增广路径p是关键的',
            '在沿增广路径对流进行增加后,该路径上的任何关键边便从残留网络中消失')
        print('  在增广路径上至少有一条边必为关键边.|E|条边中的每一条都可能至多|V|/2-1次地成为关键边')
        print('用于在用广度优先搜索寻找增广路径时,FORD-FULKERSON中的每次迭代都可以在O(E)的运行时间内完成',
            '所以Edmonds-Karp算法的全部运行时间为O(V^2E).压入与重标记算法能够达到更好的界')
        print('练习26.2-1 斜杠左边表示流,右边表示容量;流是19,容量是')
        print('练习26.2-2 Edmonds-Karp算法的执行过程')
        _fn.test_edmonds_karp()
        print('练习26.2-3 在图26-5中,对应于图中最大流的最小割是多少',
            '在例中出现的增广路径中,哪两条路径抵消了先前被传输的流')
        print('练习26.2-4 证明对任意一对顶点u和v、任意的容量和流函数c和f,有:',
            'cf(u, v)+cf(v, u)=c(u,v)+c(v,u)')
        print('练习26.2-5 通过增加具有无限容量的边,把一个多源点多汇点的流网络转换为单源点单汇点的流网络',
            '证明:如果初始的多源点多汇点网络中的边具有有限的容量,则转换后所得的网络中的任意流均为有限值')
        print('练习26.2-6 假定在多源点多汇点问题中,每个源点si产生pi单位的流,即f(si,V)=pi.同时假定每个汇点tj消耗qj单位的流',
            '即f(V,ti)=qj,其中∑pi=∑pj,说明如何把寻找一个流f以满足这些附加限制的问题,转化为在一个单源点单汇点流网络中寻找最大流的问题')
        print('练习26.2-7 证明引理26.3')
        print('练习26.2-8 证明：一个网络G=(V,E)的最大流总可以被至多由|E|条增广路径所组成的序列发现(找出最大流后再确定路径)')
        print('练习26.2-9 无向图边连通度(edge connectivity)是指为了使图不连通而必须去掉的最少边数k',
            '例如,数的边连通度为1,顶点的循环链的边连通度是2,说明如何对至多|V|个流网络',
            '(每个流网络有O(V)个顶点和O(E)条边运行最大流算法),就可以确定无向图G=(V,E)的边连通度')
        print('练习26.2-10 假定一个流网络G=(V,E)中有对称边,也就是(u,v)∈E当且仅当(v,u)∈E',
            '试证明Edmonds-Karp算法在至多进行|V||E|/4次迭代后将终止执行',
            '(对任意边(u,v),考虑(u,v)为关键边的时刻之间,d(s,u)和d(v,t)是如何变化的)')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_3:
    '''
    chpater26.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.3 note

        Example
        ====
        ```python
        Chapter26_3().note()
        ```
        '''
        print('chapter26.3 note as follow')  
        print('26.3 最大二分匹配')
        print('一些组合问题可以很容易地转换为最大流问题.26.1节中的多源点,多汇点最大流问题就是一个例子')
        print('如在一个二分图中寻找最大匹配.为了解决这一问题,将利用Ford-Fulkerson方法提供的完整性性质')
        print('可以应用Ford-Fulkerson方法在O(VE)时间内解决图G=(V,E)的最大二分匹配问题')
        print('最大二分匹配问题')
        print('  给定一个无向图G=(V,E),一个匹配(matching)是一个边的子集合,且满足对所有的顶点v∈V,M中至多一条边与v关联',
            '如果M中某条边与v关联,则说顶点v∈V被匹配,否则说v是无匹配的.最大匹配是最大势的匹配,也就是说,是满足对任意匹配M`',
            '有|M|>=|M`|的匹配M')
        print('  假定顶点集合可被划分为V=L∨R,其中L和R是不相交的,且E中的所有边的一个端点在R中,另一端点在L中',
            '进一步假设V中的每个顶点至少有一条关联的边')
        print('  二分图的最大匹配问题有着许多世纪的应用.例如,把一个机器集合L和要同时执行的任务集合R相匹配。',
            'E中有边(u,v),就说明一台特定机器u∈L能够完成一项特定任务v∈R,最大匹配可以为尽可能多的机器提供任务')
        print('寻找最大二分匹配')
        print('  利用Ford-Fulkerson方法可以在关于|V|和|E|的多项式时间内,找出无向二分图G=(V,E)的最大匹配',
            '解决这一问题的关键技巧在于建立一个流网络,其中流对应于匹配')
        print('  对二分图G的相应流网络G`=(V`,E`)定义如下,设源点s和汇点t是不属于V的新顶点')
        print('V`=V∪{s,t}.如果G的顶点划分为V=L∪R,G`的有向边为E的边,从L指向R,再加上V条新边:')
        print('E`={(s,u):u∈L}∪{(u,v):u∈L,v∈R,(u,v)∈E}∪{(v,t):v∈R}')
        print('在结束构造工作之前,在E`中的每条边赋予单位容量.因为V中的每个顶点至少有一条关联边',
            '|E|>=|V|/2.因此|E|<=|E`|=|E|+|V|<=3|E|,则|E`|=Θ(E)')
        print('引理26.10 设G=(V,E)是一个二分图,其定点划分为V=L∪R,设G`=(V`,E`)是它相应的流网络',
            '如果M是G的匹配,则G`中存在一个整数值的流f,且|f|=|M|.',
            '相反地,如果f是G`的整数值流,则G中存在一匹配M满足|M|=|f|')
        print('在一个二分图G中,一个最大匹配对应于流网络G`中的一个最大流.因此,可以通过对G`运行最大流算法来计算出G的最大匹配',
            '这一推理过程中存在的唯一故障就是最大流算法可能返回一个G`的非整数量的流f(u,v),即使流的值|f|应该为一个整数')
        print('定理26.11(完整性定理)如果容量函数c只取整数值,则由Ford-Fulkerson方法得出的最大流f满足|f|为整数的性质',
            '此外,对所有顶点u和v,f(u,v)的值为整数')
        print('推论26.12 二分图G的一个最大匹配M的势是其相应的流网络G`中的某一最大流f的值')
        print('对于一个无向二分图G,可以利用下列方法找出其最大匹配:先建立流网络G`,对它进行Ford-Fulferson方法',
            '根据求得的具有整数值的最大流f,就可直接获得最大匹配M,因为二分图上的任何匹配的势至多为min(L,R)=O(V)',
            '所以G`中最大流的值为O(V),因此,可以在O(VE`)=O(VE)的时间内,找出一个二分图的最大匹配,因为|E`|=Θ(E)')
        print('练习26.3-1 对图26-8b中所示的流网络运行Ford-Fulkerson算法.并指出每次对流增加以后所得的残留网络,对L中的顶点从上到下编为1-5号,对R中的顶点从上到下编号为6-9号',
            '在每次迭代中,找出按辞典顺序排列时,最小的一条增广路径')
        print('练习26.3-2 证明定理26.11')
        print('练习26.3-3 设G=(V,E)为二分图,其定点划分为V=L∨R,且G`是其相应的流网络',
            '在Ford-Fulkerson执行过程中,对在G`中找出的任意增广路径的长度给出一个适当的上界')
        print('练习26.3-4 完全匹配(perfect matching)是指图中每个顶点均被匹配,设G=(V,E)是其定点分划为V=L∪R的一个无向二分图',
            '其中|L|=|R|.对任意X∈V,定义X的邻居为N(X)={y∈V,(x,y)∈E,对某个x∈X}',
            '即,由与X的某元素相邻的顶点所构成的集合.证明Hall定理：G中存在一个完全匹配,当且仅当对每个子集A∈L,有|A|<=|N(A)|')
        print('练习26.3-5 在二分图G=(V,E)中,V=L∪R,如果每个顶点v∈V的度均为d,则说该图是d正则的.每个d正则二分图有|L|=|R|',
            '证明:对每个d正则二分图,其相应流网络的最小割的容量为|L|.',
            '运用该结论证明:每个d正则二分图均有一个势为|L|的匹配')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_4:
    '''
    chpater26.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.4 note

        Example
        ====
        ```python
        Chapter26_4().note()
        ```
        '''
        print('chapter26.4 note as follow')  
        print('26.4 压入与重标记算法')
        print('目前许多关旭最大流问题的渐进最快速算法就是压入与重标记算法,',
            '最大流算法最快速的实际实现都是基于压入与重标记方法的')
        print('其他有关流的问题,如最小代价流问题,也可以有效地利用压入与重标记方法来解决')
        print('Goldberg的一般性最大流算法,该算法有一种简单的实现,其运行时间为O(V^2E),',
            '这是对Edmonds-Karp算法的O(VE^2)时间的一种改进,26.5节中对一般性算法进行进一步的精化',
            '得到另外一种运行时间为O(V^3)的压入与重标记算法')
        print('相对于Ford-Fulkerson方法来说,压入与重标记采用的是一种更局部化的方法',
            '它不是检查整个残留网络G=(V,E)来找出增广路径','而是每次仅对一个顶点进行操作',
            '并且仅检查残留网络中该顶点的相邻顶点')
        print('此外,与Ford-Fulkerson方法不同,压入与重标记算法在执行过程中,并不能保持流守恒特性')
        print('但是,该算法保持了一个前置流,它是一个函数f:V*V->R,他满足反对称性、容量限制和下列放宽条件的流守恒性特性',
            '对所有顶点u∈V-{s},有f(V,u)>=0.亦即,进入除源顶点以外的顶点的总净流为非负值.进入顶点u的总净流称为进入u的余流',
            '由下式给出e(u)=f(V,u),对一个顶点u∈V-{s,t},如果e(u)>0,则称顶点u溢出')
        print('压入与重标记方法所包含的直观思想')
        print('  把一个流网络G=(V,E)看成是具有给定容量、且互相连接的管道所组成的系统',
            '把这个比方应用到Ford-Fulkerson方法中,可以说网络中的每一条增广路径均引发一条无分支点',
            '从源点到汇点的额外液体流,Ford-Fulkerson方法以迭代的方式加入更多的流,直至不能加入时为止')
        print('  从直观上看,一般性压入与重标记算法的思想在某种程度上来说有所不同.和先前一样,图的有向边对应于管道',
            '而作为管道结合点的顶点却有着两个有趣的特性.第一，为了容纳余流,每个顶点均有一个排出管道',
            '它导向能积聚液体的任意大容量水库','每个顶点和它的水库以及所有的管道连接点都处于一个平台上,当算法向前压入时,平台随之逐渐升高')
        print('  顶点的高度决定了如何压入流:仅仅把流向下压,即从较高顶点向较低顶点压',
            '从较低顶点到较高顶点可能存在一正向网络流,但是对流的压入总是向下压',
            '源点的高度固定为|V|,汇点的高度固定为0.所有其他顶点的高度开始时都是零,并逐步增加')
        print('  最终,有可能达到汇点的所有流均到达汇点。因为管道服从容量限制,',
            '并且通过任何割的流量依然受到割的容量限制,这时再没有流能到达汇点了')
        print('基本操作')
        print('  压入与重标记算法中要执行两种基本操作:把流的余量从一顶点压入到它的一个相邻顶点,以及重标记一个顶点',
            '采用这两种操作中的哪一种取决于顶点的高度.顶点的高度的准确定义：')
        print('设G=(V,E)是一个流网络,其源点为s,汇点为t.设f是G的一个前置流.如果函数h:V->N满足h(s)=|V|,h(t)=0',
            '且对每条残留边(u,v)∈Ef,有h(u)<=h(v)+1,则该函数为高度函数')
        print('引理26.13 设G=(V,E)是一个网络流,其源点为s,汇点为t.设f是G的一个前置流.h是定义在V上的高度函数',
            '对任意两顶点u`,v∈V,如果h(u)>h(v)+1,则(u,v)不是残留图中的边')
        print('压入操作')
        print('  如果u是某溢出顶点,cf(u,v)>0且h(u)=h(v)+1,则可以应用基本操作PUSH(u,v)',
            '对隐式给出的网络G=(V,E)中的前置流f进行更新,它假定对给定的c和f可以在常数时间内计算出残留容量',
            '存储于顶点u的余流用e[u]表示,u的高度用h[u]表示.符号df(u,v)是存储能够从u压入到v的流量的一个临时变量')
        print('  称PUSH(u,v)是一个从u到v的压入.如果压入操作适用于离开顶点u的某边(u,v),则也可以说压入操作适用于u',
            '如果边(u,v)变为饱和(压入后cf(u,v)=0),则该压入是饱和压入,否则就是一个不饱和压入.如果一条边是饱和的,',
            '则它不出现在残留网络中.一个简单的引理说明了不饱和压入的结果')
        print('引理26.14 在一次从u到v的不饱和压入操作后,顶点u不再溢出')
        print('重标记操作')
        print('  如果u是溢出顶点，且对所有边(u,v)∈Ef有h[u]<=h[v],则可以应用基本操作RELABEL(u).',
            '已知溢出顶点u,如果对从u到v还存在残留容量的每一个顶点v,由于v的高度不在u之下而使我们',
            '不能把流从u压入到v,则此时可以重标记溢出顶点u','源点s和汇点t都不可能是溢出顶点,因此s和t都不能被重标记')
        print('  调用操作RELABEL(u)时,说顶点u被重标记了.当u被重标记时,Ef必须至少包含一条离开u的边')
        print('  操作RELABEL(u)使u在高度函数约束下,具有所允许的最大高度')
        print('一般性算法')
        print('  在流网络中建立一个初始前置流')
        print('  INITIALIZE-FREFLOW建立的初始前置流f定义为')
        print('    f[u,v]=c(u,v); if u == s')
        print('    f[u,v]=-c(v,u); if v == s')
        print('    f[u,v]=0 else')
        print('  即每条边离开源点s的边被充满,而其他所有边不运载任何流.对每个与源点邻接的顶点v',
            '开始时有e[v]=c(s,v),e[s]被初始化为这些容量的和的负值')
        print('  一般性算法中也从高度函数h开始.h由下式给出')
        print('     h[u]=|V|; 如果u=s')
        print('     h[u]=0  else')
        print('  这是一个高度函数,因为满足h[u]>h[v]+1的边(u,v)仅是那些满足u==s的边,',
            '并且那些边是饱和的,这意味着它们不在残留网络中')
        print('引理26.15 (溢出的顶点可以被压入或重标记) 设G=(V,E)是一个流网络,源点为s',
            '汇点为t,f为一前置流。设h是f的任意高度函数.如果u是任意溢出顶点,则压入操作或重标记操作适用于该顶点')
        print('压入与重标记方法的正确性')
        print('  一般性算法压入与重标记算法解决了最大流问题,如果算法终止,则前置流f为一个最大流')
        print('引理26.16 (顶点高度不会减小) GENERIC-PUSH-RELABEL在流网路G=(V,E)上执行过程中,对每个顶点u∈V',
            '其高度h[u]至少增加1')
        print('引理26.17 设G=(V,E)是一个流网络,其源点为s,汇点为t.在GENERIC-PUSH-RELABEL对G的执行过程中,属性h始终为高度函数')
        print('引理26.18 设G=(V,E)是一个流网络,其源点为s,汇点为t.设f是G的一前置流,h是定义在V上的高度函数',
            '则在残留网络Gf中不存在从源点s到汇点t的路径')
        print('定理26.19(一般性压入与重标记算法的正确性) 当算法GENERIC-PUSH-RELABEL在具有源点s和汇点t的流网络G=(V,E)上运行时',
            '若算法终止,则它计算出的前置流f为G的最大流')
        print('对压入与重标记方法的分析')
        print('  对于三种类型的操作(重标记,饱和压入和不饱和压入)中的每一种,分别给出其界',
            '有了这些界后,构造一个运行时间为O(V^2E)的算法就成为一个简单的问题了')
        print('引理26.20 设G=(V,E)是源点为s,汇点为t的一个流网络,且f是G的前置流.',
            '则对任意溢出顶点u,在残留网络Gf中存在着一条从u到s的简单路径')
        print('引理26.21 设G=(V,E)是一个网络流,其源点为s,汇点为t.GENERIC-PUSH-RELABEL在G上执行过程中的任何时刻',
            '对所有的顶点u∈V,都有h[u]<=2|V|-1')
        print('  证明:因为根据定义,源点s和汇点t均不是溢出顶点,所以它们的高度不会改变.',
            '因此总是有h[s]=|V|和h[t]=0,都不会大于2|V|-1')
        print('推论26.22 (关于重标记操作的界) 设G=(V,E)是一个流网络,其源点为s,汇点为t.',
            '在GENERIC-PUSH-RELABEL对G的执行过程中,对每个顶点执行重标记操作的次数之多为2|V|-1',
            '全部重标记操作的执行次数至多为(2|V|-1)(|V|-2)<2|V|^2')
        print('引理26.23 (关于饱和压入的界) 在GENERIC-PUSH-RELABEL对任意流网络G=(V,E)的执行过程中,',
            '饱和压入的次数至多为2|V||E|')
        print('引理26.24 (关于不饱和压入的界) 在GENERIC-PUSH-RELABEL对任意流网络G=(V,E)的执行过程中',
            '不饱和压入的次数至多为:4|V|^2(|V|+|E|)')
        print('定理26.25 在GENERIC-PUSH-RELABEL对任意流网络G=(V,E)的执行过程中,基本操作的执行次数为O(V^2E)')
        print('推论26.26 对任意流网络G=(V,E),存在一种压入与重标记算法的实现,其运行时间为O(V^2E)')
        print('练习26.4-1 说明如何实现一般性压入与重标记算法,使得每次重标记操作需要O(V)时间,每次压入操作需要O(1)时间',
            '选择可应用的操作需要O(1)时间,而整个算法的时间为O(V^2E)')
        print('练习26.4-2 证明：在执行全部的O(V^2)次重标记操作,一般性压入与重标记算法所需的全部运行时间仅为O(VE)')
        print('练习26.4-3 假定运用压入与重标记算法找出了流网络G=(V,E)中的最大流.',
            '试给出一种快速算法来找出G的最小割')
        print('练习26.4-4 写出一个有效的压入与重标记算法,以找出一个二分图的最大匹配')
        print('练习26.4-5 假定在流网络G=(V,E)中,所有边的容量都属于集合{1,2,...,k}.',
            '试用|V|,|E|和k来分析一般性压入与重标记算法的时间(一条边在成为饱和边前能承受多少次不饱和压入)')
        print('练习26.4-6 证明INITIALIZE-PREFLOW的第7行可以改为h[s]<-|V[G]|-2,',
            '这并不影响一般性压入与重标记算法的正确性或其渐进意义上的性能')
        print('练习26.4-7 设df(u,v)为残留网络Gf中从u到v的距离(边数).证明GENERIC-PUSH-RELABEL保持下列性质:',
            '若h[u]<|V|,则h[u]<=df(u,t);并且h[u]>=|V|,则h[u]-|V|<=df(u,s)')
        print('练习26.4-8 如上一个练习题中一样,设df(u,v)为残留网络Gf中从u到v的距离.',
            '说明如何修改一般性压入与重标记算法以保持下列性质:')
        print('  若h[u]<|V|,则h[u]=df(u,t);并且若h[u]>=|V|,则h[u]-|V|=df(u,s)',
            '为保持这一性质,算法的全部运行时间为O(VE)')
        print('练习26.4-9 证明对于|V|>=4,在流网络G=(V,E)上运行GENERIC-PUSH-RELABEL',
            '所执行的非饱和压入操作的次数至多为4|V|^2|E|')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

class Chapter26_5:
    '''
    chpater26.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter26.5 note

        Example
        ====
        ```python
        Chapter26_5().note()
        ```
        '''
        print('chapter26.5 note as follow')  
        print('26.5 重标记与前移算法')
        print('压入与重标记方法允许以任意次序执行基本操作.但是,通过仔细选择执行次序和有效安排网络的数据结构',
            '可以用比推论26.26给出的O(V^2E)更少的运行时间来解决最大流问题.')
        print('重标记与前移算法,这是一种运行时间为O(V^3)的压入与重标记算法,',
            '从渐进意义上来看,对于稠密网络它至少不弱于O(V^2E)')
        print('重标记与前移算法设置了一张网络中顶点表,算法从表的前端开始扫描表,反复选出溢出顶点u,然后\"排除\"它,',
            '即反复执行压入和重标记操作,直至顶点u不再存在正的余流,当某个顶点被重标记时,就被移动到表的前端(所以算法名为\"重标记与前移\")',
            '算法又重新开始扫描')
        print('重标记算法与前移算法的正确性及其性能分析与概念\"容许\"边(admissible edge)有关',
            '即残留网络中压入的流经过的那些边.在证明关于容许网络的几条性质后,将讨论排出操作,最后给出并分析重标记与前移算法')
        print('容许边和容许网络')
        print('  设G=(V,E)是一个流网络,其源点为s,汇点为t.f是G的前置流,h是高度函数,则如果cf(u,v)>0',
            '且h(u)=h(v)+1,就说(u,v)是容许边.否则，(u,v)是非容许边.容许网络为Gf,h=(V,Ef,h),其中Ef,h为容许边的集合')
        print('引理26.27 (容许网络中不包含回路) 如果G=(V,E)是一个流网络,f是G的一个前置流,且h是G上的高度函数,',
            '则容许网络Gf,h=(V,Ef,h)中不包含回路')
        print('引理26.28 设G=(V,E)是一个流网络,f是G的一个前置流,且h是G上的高度函数.',
            '如果顶点u是溢出顶点且(u,v)是容许边,则采用PUSH(u,v).该操作不会建立任何新的容许边,但它可能使(u,v)变为非容许边')
        print('引理26.29 设G=(V,E)是一个流网络,f是G的一个前置流,且h是G上的高度函数.如果顶点u是溢出顶点,并且不存在在离开u的容许边',
            '则此时RELABEL(u)适用.在执行重标记操作后,至少存在一条离开u的容许边,但不会进入u的容许边')
        print('相邻表')
        print('  重标记与前移算法中的边都被放入\"相邻表\"中.如果给定流网络G=(V,E),',
            '对顶点u∈V,其相邻表N[u]是一个关于G中u的相邻顶点的单链表')
        print('  因此,如果(u,v)∈E或(v,u)∈E,则顶点v出现在表N[u]中.相邻表N[u]仅仅包含哪些可能存在残留边(u,v)的顶点v',
            'N[u]中的第一个顶点由head[N[u]]指出。相邻表中v的下一个顶点由指针next-neighbor[v]指出；',
            '如果v是相邻表中的最后一个顶点,则该指针为None')
        print('  重标记与前移算法在执行过程中,按某确定的顺序循环访问每个相邻表.对每个顶点u,',
            '域current[u]指向N[u]中当前被考察的顶点.current[u]开始时被置为head[N[u]]')
        print('溢出顶点的排除')
        print('  一个溢出顶点u通过下列方式排除：把该顶点的所有余流通过容许边压入到相邻顶点',
            '必要时重标记顶点u,使离开u的边变成容许边')
        print('重标记与前移算法')
        print('  设置包含V-{s,t}中所有顶点的链表L.',
            '该链表的一个重要性质是根据容许网络对表中的所有顶点进行拓扑排序')
        print('  容许网络是一个有向无回路图')
        print('在重标记与前移算法伪代码中,假设对每个顶点u已经建立了相邻表N[u]',
            '并假定next[u]指向L中u的后继顶点.若是表中的最后一个顶点,则next[u]=None')
        print('定理26.31 RELABEL-TO-FRONT对任意流网络G=(V,E)的运行时间为O(V^3)')
        print('  RELABEL-TO-FRONT的运行时间为O(V^3+VE)=O(V^3)')
        print('练习26.5-1 假定L中的初始顶点顺序<v1,v2,v3,v4>,相邻表为',
            'N[v1]=<s,v2,v3>',
            'N[v2]=<s,v1,v3,v4>',
            'N[v3]=<v1,v2,v4,t>',
            'N[v4]=<v2,v3,t>',
            '说明RELABEL-TO-FRONT对图的执行过程')
        print('练习26.5-2 希望通过对溢出顶点设置一个先进先出队列的方法来实现压入与重标记算法',
            '算法反复排除处于队头的顶点,任何在排除前不为溢出但排除后变为溢出的顶点被放在队列末尾',
            '当队头的顶点被排除后,就把它从队列中去掉.当队列为空时,算法终止',
            '证明可以实现这一算法,使其能在O(V^3)的时间内计算出最大流')
        print('练习26.5-3 证明,如果RELABEL仅通过计算h[u]<-h[u]+1来更新h[u],一般性算法依然正确',
            '这一变化对RELABEL-TO-FRONT的性能有何影响')
        print('练习26.5-4 证明:如果总是排除最高的溢出顶点,则可以使压入与重标记算法的运行时间变为O(V^3)')
        print('练习26.5-5 假定在压入与重标记算法执行的某一时刻,存在一个整数0<k<=|V|-1,没有顶点满足h[v]=k',
            '证明所有h[v]>k的顶点都在最小割的源点一边.如果这样的k存在,间隙启发式(gap heuristic)对h[v]>k的v∈V-s',
            '的每个顶点v进行更新,置h[v]<-max(h[v],|V|+1).证明所得到的属性h为高度函数',
            '间隙启发式对压入与重标记算法的良好运行来说非常关键')
        print('思考题26-1 逃脱问题')
        print('  一个n*n栅格是由n行和n列顶点组成的一个无向图,用(i,j)表示处于第i行第j列的顶点',
            '除了边界顶点(即满足i=1,i=n,j=1或j=n的顶点(i,j)),栅格中的所有其他顶点都有四个相邻顶点')
        print('  逃脱问题即确定从起始顶点到边界上的任何m个相异的顶点之间,是否存在m条其定点不相交的路径')
        print('思考题26-2 最小路径覆盖')
        print('  在有向图G=(V,E)中,路径覆盖是一个其顶点不相交的路径的集合P,满足V中的每一个顶点',
            '仅包含于P中的一条路径中.路径可以从任意顶点开始和结束.且长度也为任意值,包括0.',
            'G的一个最小路径覆盖是指包含尽可能少的路径的路径覆盖')
        print('思考题26-3 航天飞机实验')
        print('  网络G包含一个源点顶点s,顶点I1,I2,...,In,顶点E1,E2,...,Em和一个汇点顶点t.对k=1,2,...,n',
            '存在一条容量为ck的边(s,Ik),且对j=1,2,...,m,存在一条容量为pj的边(Ej,t).对k=1,2,...,n',
            '和i=1,2,...,m,如果Ik∈Rj,则存在一条无限容量的边(Ik,Ej)')
        print('思考题26-4 最大流的更新')
        print('  设G=(V,E)是源点s,汇点为t并且具有整数数量的一个流网络.假定已知G中的一个最大流',
            'a) 假定把一条边(u,v)∈E的容量增加1.写出一个运行时间为O(V+E)的算法以更新最大流',
            'b) 假定把一条边(u,v)∈E的容量减小1.试写出一个运行时间为O(V+E)的算法以更新最大流')
        print('思考题26-5 用定标法计算最大流')
        print('  设G=(V,E)是源点为s,汇点为t的一个流网络.其每条边(u,v)∈E的容量c(u,v)为整数,设C=maxc(u,v)')
        print('  证明:对一给定的数K,如果存在一条容量至少为K的增广路径,则可以在O(E)的时间内找出该路径')
        print('思考题26-6 具有负容量的最大流')
        print('  假定允许一个流网络有负容量的边(可以有正容量的边).在这样的网络中,可以不存在可行流')
        print('  a) 考虑流网络G=(V,E)中c(u,v)<0,用u和v之间的流来简单地解释负容量的含义')
        print('  b) 如果G中存在一个可行流,则G`中的所有容量均为非负值,而且G`中存在一个最大流,其所有进入汇点t`的边都饱和了')
        print('  c) 给定G`中的流,进入t`的边都饱和,需要说明如何得到G中的一个可行流')
        print('思考题26-7 Hopcroft-Karp二分图匹配算法')
        print('  Hopcroft-Karp二分图匹配算法的运行时间为O(sqrt(V)E).给定一个无向二分图G=(V,E),其中V=L∪R且所有边仅有一个端点在L中',
            '设M为G的一个匹配.称简单路径P为G中关于M的增广路径,如果此路径从L中未匹配顶点出发,终止于R中的未匹配顶点,并且路径上的边交替地属于M和E-M')
        print('')
        # python src/chapter26/chapter26note.py
        # python3 src/chapter26/chapter26note.py

chapter26_1 = Chapter26_1()
chapter26_2 = Chapter26_2()
chapter26_3 = Chapter26_3()
chapter26_4 = Chapter26_4()
chapter26_5 = Chapter26_5()

def printchapter26note():
    '''
    print chapter26 note.
    '''
    print('Run main : single chapter twenty-six!')  
    chapter26_1.note()
    chapter26_2.note()
    chapter26_3.note()
    chapter26_4.note()
    chapter26_5.note()

# python src/chapter26/chapter26note.py
# python3 src/chapter26/chapter26note.py
if __name__ == '__main__':  
    printchapter26note()
else:
    pass

```

```py
"""
module flownetwork
===

contains alrotithm with max flow

"""
import math as _math
from copy import deepcopy as _deepcopy

from graph import * 

class _FlowNetwork:
    '''
    流网络相关算法集合类
    '''
    def __init__(self, *args, **kwargs):
        '''
        流网络相关算法集合类
        '''
        pass

    def _findbfs(self):
        '''
        广度搜索算法寻找是否存在增广路径`p`
        '''
        return False

    def ford_fulkerson(self, g : Graph, s : Vertex, t : Vertex):
        '''
        基本的`Ford-Fulkerson`算法
        '''
        for edge in g.edges:
            edge.flowtofrom = 0
            edge.flowfromto = 0
    
    def edmonds_karp(self, g : Graph, s : Vertex, t : Vertex):
        '''
        使用广度优先搜索实现增广路径`p`计算的`Edmonds-Karp`算法
        '''
        for edge in g.edges:
            edge.flowtofrom = 0
            edge.flowfromto = 0

    def relabel(self, u :Vertex):
        '''
        重标记算法 标记顶点`u`
        '''
        pass
    
    def initialize_preflow(self, g : Graph, s : Vertex):
        '''
        一般性压入与重标记算法

        Args
        ===
        `g` : 图`G=(V,E)`

        `s` : 源顶点`s`

        '''
        for u in g.veterxs:
            u.h = 0
            u.e = 0
        for edge in g.edges:
            edge.flowfromto = 0
            edge.flowtofrom = 0
        s.h = g.vertex_num
        adj = g.getvertexadj(s)
        for u in adj:
            edge = g.getedge(s, u)
            edge.flowfromto = edge.capcity
            edge.flowtofrom = -edge.capcity
            u.e = edge.capcity
            s.e = s.e - edge.capcity

    def generic_push_relabel(self, g : Graph, s : Vertex):
        '''
        基本压入重标记算法
        '''
        self.initialize_preflow(g, s)
    
    def push(self, u : Vertex, v : Vertex):
        '''
        压入算法
        '''
        pass

    def discharge(self, g : Graph, u : Vertex):
        '''
        溢出顶点`u`的排除
        '''
        while u.e > 0:
            v = u.current
            if v == None:
                self.relabel(u)
                # head[N]
                u.current = u.N[0]
            elif g.getedge(u, v).flowfromto > 0 and u.h == v.h + 1:
                self.push(u, v)
            else:
                u.current = v.next_neighbor

    def relabel_to_front(self, g : Graph, s : Vertex, t : Vertex):
        '''
        重标记与前移算法 时间复杂度`O(V^3)`
        '''
        self.initialize_preflow(g, s)
        L = topological_sort(g)
        for u in g.veterxs:
            # head[N]
            u.current = u.N[0]
        index = 0
        while index < len(L):
            u = L[index]
            old_height = u.h
            self.discharge(g, u)
            if u.h > old_height:
                q = L.pop(index)
                L.insert(0, q)
            index += 1

__fn_instance = _FlowNetwork()

ford_fulkerson = __fn_instance.ford_fulkerson
edmonds_karp = __fn_instance.edmonds_karp
generic_push_relabel = __fn_instance.generic_push_relabel

def _buildtestgraph():
    '''
    构造测试有向图`G=(V,E)`
    '''
    g = Graph()
    vertexs = ['s', 't', 'v1', 'v2', 'v3', 'v4']
    g.addvertex(vertexs)
    g.addedgewithdir('s', 'v1', 16)
    g.addedgewithdir('s', 'v2', 13)
    g.addedgewithdir('v1', 'v2', 10)
    g.addedgewithdir('v2', 'v1', 4)
    g.addedgewithdir('v1', 'v3', 12)
    g.addedgewithdir('v2', 'v4', 14)
    g.addedgewithdir('v3', 'v2', 9)
    g.addedgewithdir('v3', 't', 20)
    g.addedgewithdir('v4', 't', 4)
    g.addedgewithdir('v4', 'v3', 7)
    return g

def test_ford_fulkerson():
    '''
    测试基本的`Ford-Fulkerson`算法
    '''
    g = _buildtestgraph()
    print('邻接矩阵为')
    print(g.getmatrixwithweight())

def test_edmonds_karp():
    '''
    测试`Edmonds-Karp`算法
    '''
    g = _buildtestgraph()
    print('邻接矩阵为')
    print(g.getmatrixwithweight())

def test():
    '''
    测试函数
    '''
    test_ford_fulkerson()
    test_edmonds_karp()

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter27/chapter27note.py
# python3 src/chapter27/chapter27note.py
'''

Class Chapter27_1

Class Chapter27_2

Class Chapter27_3

Class Chapter27_4

Class Chapter27_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    pass
else:
    pass

class Chapter27_1:
    '''
    chpater27.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.1 note

        Example
        ====
        ```python
        Chapter27_1().note()
        ```
        '''
        print('chapter27.1 note as follow')  
        print('第七部分 算法研究问题选编')
        print('第27章给出一种并行计算模型,即比较网络.比较网络是允许同时进行很多比较的一种算法',
            '可以建立比较网络,使其在O(lg^2n)运行时间内对n个数进行排序')
        print('第28章研究矩阵操作的高效算法,通过考察矩阵的一些基本性质,讨论Strassen算法',
            '可以在O(n^2.81)时间内将两个n*n矩阵相乘.然后给出两种通用算法,即LU分解和LUP分解,',
            '在利用高斯消去法在O(n^3)时间内解线性方程时要用到这两种方法',
            '当一组线性方程没有精确解时,如何计算最小二乘近似解')
        print('第29章研究线性规划.在给定资源限制和竞争限制下,希望得到最大或最小的目标',
            '线性规划产生于多种实践应用领域.单纯形法')
        print('第30章 快速傅里叶变换FFT,用于在O(nlgn)运行时间内计算两个n次多项式的乘积')
        print('第31章 数论的算法：最大公因数的欧几里得算法；',
            '求解模运算的线性方程组解法，求解一个数的幂对另一个数的模的算法',
            'RSA公用密钥加密系统，Miller-Rabin随机算法素数测试,有效地找出大的素数；整数分解因数')
        print('第32章 在一段给定的正文字符串中，找出给定模式的字符串的全部出现位置')
        print('第33章 计算几何学')
        print('第34章 NP完全问题')
        print('第35章 运用近似算法有效地找出NP完全问题的近似解')
        print('第27章 排序网络')
        print('串行计算机(RAM计算机)上的排序算法,这类计算机每次只能执行一个操作',
            '本章中所讨论的排序算法基于计算的一种比较网络模型','这种网络模型中,可以同时执行多个比较操作')
        print('比较网络与RAM的区别主要在于两个方面.前者只能执行比较,因此,像计数排序这样的算法就不能在比较网络上实现',
            '其次,在RAM模型中,各操作是串行执行的,即一个操作紧接着另一个操作')
        print('在比较玩过中,操作可以同时发生,或者以并行方式发生,这一特点使得我们能够构造出一种在次线性的运行时间内对n个值进行排序的比较网络')
        print('27.1 比较网络')
        print('排序网络总是能对其他输入进行排序的比较网络,比较网络仅由线路和比较器构成',
            '比较器是具有两个输入x和y以及两个输出x`和y`的一个装置,它执行下列函数')
        print('假设每个比较器操作占用的时间为O(1),换句话说,假定出现输入值x和y与产生输出值x`和y`之间的时间为常数')
        print('一条线路把一个值从一处传输到另一处,可以把一个比较器的输出端与另一个比较器的输入端相连',
            '在其他情况下,它要么是网络的输入线,要么是网络的输出线.',
            '在本章中都假定比较网络含n条输入线以及n条输出线')
        print('只有当同时有两个输入时,比较器才能产生输出值.假设在时间0输入线路上出现了一个输入序列<9,5,2,6>',
            '则在时刻0，只有比较器A和B同时存在两个输入值.假定每个比较器要花1个单位的时间来计算出输出值')
        print('在每个比较器均运行单位时间的假设下,可以对比较网络的\"运行时间\"作出定义',
            '就是从输入线路接受到其值的时刻,到所有输出线路收到其值所花费的时间.',
            '非形式地说,这一运行时间就是任何输入元素从输入线路到输出所经过的比较器数目的最大值')
        print('一条线路的深度可以定义：比较网络的输入线路深度为0.如果一个比较器有两条深度分别为dx和dy的输入线路',
            '则其输出线路的深度为max(dx+dy)+1')
        print('由于比较网络中没有比较器回路,所以线路的深度有明确定义,并且定义比较器的深度为其输出线路的深度')
        print('排序网络是指对每个输入序列,其输出序列均为单调递增(即b1<=b2<=...<=bn)的一种比较网络')
        print('比较网络与过程的相似之处在于它指定如何进行比较,其不同之处在于其实际规模决定于输入和输出的数目')
        print('练习27.1-1 给定一输入序列<9 6 5 2>,说明图上网络所有线路出现的值')
        print('练习27.1-2 设n为2的幂，试说明如何构造一个具有n个输入和n个输出，且深度为lgn的比较网络，',
            '其顶部的输出线路总是输出最小的输入值，而底部的输出线路则总是输出最大的输入值')
        print('练习27.1-3 向一个比较器后,所得的比较网络可能不再是排序网络了')
        print('练习27.1-4 证明任何具有n个输入的排序网路的深度至少为lgn')
        print('练习27.1-5 证明任何排序网络中的比较器的数目至少为Ω(nlgn)')
        print('练习27.1-6 说明排序网络的结构与插入排序有何关系')
        print('练习27.1-7 可以把C个比较器和n个输入的比较网络表示为取值范围从1到n,c对整数组成的一张表',
            '如果两对整数中包含同一整数,则在网络中相应的比较器排序由整数对的次序决定',
            '并描述一个运行时间为O(n+C)的串行算法来计算比较网络的深度')
        print('练习27.1-8 颠倒型比较器,这种比较器在其底部线路中产生最大输出值',
            '试说明如何把c个标准或颠倒的比较器组成的任意网络,转换为仅包含c个标准比较器的排序网络,证明所给出的转换方法是正确的')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_2:
    '''
    chpater27.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.2 note

        Example
        ====
        ```python
        Chapter27_2().note()
        ```
        '''
        print('chapter27.2 note as follow')  
        print('27.2 0-1原理')
        print('0-1原理认为，对于属于集合{0,1}的每个输入值,排序网络都能正确运行,则对任意输入值,它也能正确运行',
            '当构造排序网络和其他比较网络时,0-1原理使把注意力集中于对由0和1时组成的输入序列进行相应的操作',
            '一旦构造好排序网络,并证明它能对所有的0-1序列进行排序时,就可以运用0-1原理,说明他能对任意值序列进行正确的排序')
        print('引理27.1 如果比较网络把输入序列a=<a1,a2,a3,...,an>转化为输入序列b=<b1,b2,...,bn>',
            '则对任意单调递增函数f，该网络把输入序列f(a)=<f(a1),f(a2),...,f(an)>,转化为输出序列',
            'f(b)=<f(b1),f(b2),...,f(bn)>')
        print('对一般的比较网络中每条线路的深度进行归纳，从而证明一个比上述引理更强的结论：',
            '当把序列a作为网络的输入时，如果每条线路的值为ai，则把序列f(a)作为网络的输入时该线路的值为f(ai)',
            '因为输出线路包含于上述结论中，所以证明了该结论，也就证明了引理')
        print('定理27.2(0-1原理) 如果一个具有n个输入的比较网络能够对所有可能存在的2^n个0和1组成的序列进行正确的排序',
            '则对所有任意数组成的序列,该比较网络也可能对其正确的排序')
        print('练习27.2-1 证明：把一个单调递增函数作用于一个已排序序列后，得到的仍然是一个排序序列')
        print('练习27.2-2 证明：当且仅当能正确地对如下n-1个0-1序列进行排序：<1,0,0,...,0,0>,<1,1,0,...,0,0>',
            ',...,<1,1,1,...,1,0>,具有n个输入的比较网络才能够正确地对输入序列<n,n-1,...,1>进行排序')
        print('练习27.2-3 运用0-1原则，证明图27-6所示的比较网络为一个排序网络')
        print('练习27.2-4 对判定树模型(decision-tree model)阐述并证明与0-1原理类似的结论(提示：要正确地处理等式)')
        print('练习27.2-5 证明：对所有i=1,2,...,n-1,在一个具有n个输入的排序网络中,第i条线与第i+1条线之间必至少有一个比较器')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_3:
    '''
    chpater27.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.3 note

        Example
        ====
        ```python
        Chapter27_3().note()
        ```
        '''
        print('chapter27.3 note as follow')  
        print('27.3 双调排序网络')
        print('要构造有效的排序网络，第一步是构造一个能对任意双调序列(bitonic sequence)进行的比较网络')
        print('双调序列是指序列要么先单调递增后再单调递减，或者循环移动成为先单调递增后再单调递减')
        print('例如序列<1,4,6,8,3,2>,<6,9,4,2,3,5>和<9,8,3,2,4,6>都是双调的')
        print('对于边界情况,任何一个1个和2个数的序列都是双调序列。双调的0-1序列的结构比较简单,',
            '其形式为0^i 1^j 0^k或1^i 0^j 1^k,其中i,j,k>=0.必须注意单调递增或单调递减的序列也是单调的')
        print('将要构造的双调排序程序是一个能对0和1的双调序列进行排序的比较网络',
            '双调排序程序可以对任意数组成的双调序列进行排序')
        print('半清洁器')
        print('  双调排序由一些阶段组成,其中每一个阶段称为一个半清洁器(half-cleaner).',
            '每个半清洁器是一个深度为1的比较网络,其中输入线i与输出线i+n/2进行比较,i=1,2,...,n/2(假设n为偶数)')
        print('  当由0和1组成的双调序列作用于半清洁器输入时,半清洁器产生一个满足下列条件的输出序列:较小的值位于输出的上半部,较大的值位于输出的下半部',
            '并且两部分序列仍然是双调的。')
        print('  事实上，两部分序列中至少有一部分是清洁的--全由0或全由1组成。正是由于这一性质，才称其为\"半清洁器\",')
        print('引理27.3 如果半清洁器的输入是一个由0和1组成的双调序列，则其输出满足如下性质：输出的上半部分与下半部分都是双调的',
            '上半部分输出的每一个元素与下半部分输出的每个元素一样小,并且两部分中至少有一个部分是清洁的')
        print('双调排序器')
        print('  通过递归地连接半清洁器，就可以建立一个双调排序器，它是一个对双调序列进行排序的网络。',
            'BITONIC-SORTER[n]的第一个阶段由HALF-CLEANER[n]组成.由引理27.3可知,HALF-CLEANER[n]产生两个规模缩小一半的双调序列,',
            '且满足上半部分的每个元素至少与下半部分的每个元素一样小。因此，可以运用两个BITONIC-SORTER[n/2]分别对两部分递归地进行排序,从而完成整个排序工作')
        print('  BITONIC-SORTER[n]的深度D(n)由下列递归式给出:')
        print('  D(n)=0 if n == 1; D(n)=D(n/2)+1 if n == 2^k and k >= 1')
        print('  可以推得其解为D(n)=lgn')
        print('因此，可以用BITONIC-SORTER对深度为lgn的0-1双调序列进行排序','由类似于0-1原理的结论可知：',
            '该网络能对由任意数组成的双调序列进行排序')
        print('练习27.3-1 n=1,存在1个;n=2时存在2个;n=3时存在2个;n=4时存在6个;n=5时存在12个')
        print('  结论存在m个由0和1组成的双调序列 m=n if n <= 2; m=(n-1)(n-2) if n >= 3')
        print('练习27.3-2 证明当n为2的幂时,BITONIC-SORTER[n]包含Θ(nlgn)个比较器')
        print('练习27.3-3 说明当输入数n不是2的幂时,如何构造一个深度为O(lgn)的双调排序器')
        print('练习27.3-4 如果某半清洁器的输入是一个由任意数组成的双调序列,证明输出端满足下列性质:',
            '输出的上半部分和下半部分都是双调的,上半部分中的每个元素至少与下半部分中的每个元素一样小')
        print('练习27.3-5 考察两个由0和1组成的序列.证明如果其中一个序列的每个元素至少和另一个序列中每个元素一样小,则两个序列中有一个序列是清洁的')
        print('练习27.3-6 证明与0-1原则类似的关于双调排序网络的结论：一个能对任何0和1组成的双调序列进行排序的比较网络,也能对任何由任意数字组成双调序列进行排序')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_4:
    '''
    chpater27.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.4 note

        Example
        ====
        ```python
        Chapter27_4().note()
        ```
        '''
        print('chapter27.4 note as follow')  
        print('27.4 合并网络')
        # !合并网络就是指能把两个已排序的输入序列合并为一个有序的输出序列的网络
        print('合并网络就是指能把两个已排序的输入序列合并为一个有序的输出序列的网络')
        print('将对BITONIC-SORTER[n]加以修改，以生成合并网络MERGER[n]')
        print('和双调排序类似，下面仅对输入为0-1序列的情况来证明合并网络的正确性')
        print('合并网络基于下列直觉思想：已知两个有序序列，如果把第二个序列的顺序颠倒，',
            '再把序列连接在一起，所得序列应该为双调序列')
        print('已知两个有序的0-1序列：X=00000111和Y=00001111,把Y的顺序颠倒，得Yr=11110000',
            '再把X和Yr相连就得到双调序列0000011111110000,因此要合并两个输入序列X和Y',
            '只要对X和Yr连接成的序列执行双调排序就可以了')
        print('练习27.4-1 对于合并网络，证明一个与0-1原则类似的结论。特别地，证明一个能对任何两个由0',
            '和1组成的单调递增序列进行合并的比较网络,也能对任何两个任意数组成的单调递增序列进行合并')
        print('练习27.4-2 要把多少个不同的0-1输入序列作为一个比较网络的输入，才能验证该网络是一个合并网络')
        print('练习27.4-3 证明：对任何能把1与n-1项合并，以产生一个长度为n的排序序列的网络')
        print('练习27.4-4 考察一个输入为a1,a2,...,an的合并网络,n为2的幂,其中包含两个需要合并的单调序列',
            '<a1,a3,...,an-1>和<a2,a4,...,an>.证明：在这种合并网络中,比较器的数目为Ω(nlgn)',
            '提示：将比较器分解为3个集合')
        print('练习27.4-5 证明：不论次序如何，任何合并网络都需要Ω(nlgn)个比较器')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_5:
    '''
    chpater27.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.5 note

        Example
        ====
        ```python
        Chapter27_5().note()
        ```
        '''
        print('chapter27.5 note as follow')  
        print('27.5 排序网络')
        print('排序网络SORTER[n]运用合并网络，实现对合并排序算法的并行化')
        print('SORTER[n]的递归构造，给定n个输入元素，用两个SORTER[n/2],递归地对两个长度为n/2的子序列(并行地)进行排序',
            '然后，再用MERGER[n]对得到的两个序列进行合并。递归的边界情况是n=1;此时',
            '可以只用一条线路来对1个元素组成的序列进行排序,因为1个元素的序列已排序好了')
        print('在网络SORTER[n]中，数据要通过lgn个阶段.网络的每一个独立的输入已经是由一个元素组成的一个有序序列',
            'SORTER[n]的第一个阶段包含n/2个MERGER[2],并行地对每对由1个元素组成的序列进行合并',
            '以产生长度为2的排序序列。第二个阶段包含n/4个MERGER[4],把每对由2个元素组成的排序序列进行合并',
            '以产生长度为4的排序序列。一般来说,对于k=1,2,...,lgn,第k个阶段包含n/2^k个MERGER[2^k]',
            '把每对由2^(k-1)个元素组成的排序序列进行合并,结果是长度为2^k的排序序列',
            '在最后一个阶段,只产生由全部输入值组成的一个排序序列.可以用归纳法来证明这一排序网络能对0-1许雷进行排序',
            '因此由0-1原则可知，也能同样对任意输入值进行排序')
        print('递归地分析排序网络的深度')
        print('  SORTER[n]的深度D(n)就是SORTER[n/2]的深度D(n/2)(存在两个相同的SORTER[n/2],并行地操作)加上MERGER[n]的深度lgn',
            '因此,SORTER[n]的深度可由下列递归式定义')
        print('  D(n)=0 if n == 1; D(n)=D(n/2)+lgn if n == 2^k and k >= 1')
        print('可以推出其解为D(n)=Θ(lg^2(n)),因此可以在O(lg^2(n))的时间内并行地对n个数进行排序')
        print('练习27.5-1 SORTER[n]中有nlgn个比较器')
        print('练习27.5-2 SORTER[n]的深度恰好为(lgn)(lgn + 1)/2')
        print('练习27.5-3 假定有2^n个元素<a1,a2,...,a2n>,希望把该序列划分成为两个序列,其中一个包含n个最小值,另一个包含n个最大值',
            '证明:分别对序列<a1,a2,...,an>和<an+1,an+2,...,a2n>进行排序后再一定的深度内就可达到上述要求')
        print('练习27.5-4 设S(k)为具有k个输入的排序网络的深度,M(k)为具有2k个输入的合并网络的深度.',
            '假定对一个由n个数组成的序列进行排序,并且已知每个数与其在结果序列中的正确位置相差不超过k个数的位置',
            '证明：能够在深度S(k)+2M(k)内对这n个数进行排序')
        print('练习27.5-5 可以通过反复执行下列过程k次,来对一个m*m矩阵中的元素进行排序:')
        print('  1.每个奇数行的元素排序列单调递增序列')
        print('  2.每个偶数行的元素排序列单调递减序列')
        print('  3.把每列元素排序成单调序列')
        print('思考题27-1 排序网络的转置')
        print('  在一个比较网络中,如果每一个比较器仅连接相邻的两根线,则称这种网络为转置网络')
        print('  a) 证明：任何具有n个输入的转置网络都包括Ω(n^2)个比较器')
        print('  b) 证明：当且仅当能对序列<n,n-1,...,1>进行排序时,具有n个输入的转置网络为排序网络')
        print('思考题27-2 Batcher奇偶合并网络')
        print('  假设n为2的幂,对排序序列<a1,a2,...,an>和<an+1,an+2,...,a2n>进行合并',
            '如果n==1,在线a1和a2之间放置一个比较器,否则就递归地构造两个并行操作的奇偶合并网络',
            '第一个合并对线路上的序列<a1,a3,...,an-1>和序列<an+1,an+3,...,a2n-1>(序号为奇数的元素)进行合并',
            '第二个合并网络把序列<a2,a4,...,an>与序列<an+2,an+4,...,a2n>(序号为偶数的元素)进行合并',
            '为使两个排序序列相连接,把一个比较器放在a2i和a2i+1之间,i=1,2,...,n-1')
        print('思考题27-3 排列网络')
        print('  具有n个输入和n个输出的排列网络存在着一些开关，用来根据n!中可能的排列,把网络的输入和输出进行各种可能的连接')
        print('  证明：如果把排序网络中的每一个比较器换成图所示的开关，所得的网络就是一个排列网络,对任意排列pi,网络中存在一种置开关的方式可以使输入i与输出pi(i)相连')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

chapter27_1 = Chapter27_1()
chapter27_2 = Chapter27_2()
chapter27_3 = Chapter27_3()
chapter27_4 = Chapter27_4()
chapter27_5 = Chapter27_5()

def printchapter27note():
    '''
    print chapter27 note.
    '''
    print('Run main : single chapter twenty-seven!')  
    chapter27_1.note()
    chapter27_2.note()
    chapter27_3.note()
    chapter27_4.note()
    chapter27_5.note()

# python src/chapter27/chapter27note.py
# python3 src/chapter27/chapter27note.py
if __name__ == '__main__':  
    printchapter27note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter28/chapter28note.py
# python3 src/chapter28/chapter28note.py
"""

Class Chapter28_1

Class Chapter28_2

Class Chapter28_3

Class Chapter28_4

Class Chapter28_5

"""
from __future__ import absolute_import, division, print_function

import numpy as np

class Chapter28_1:
    """
    chapter28.1 note and function
    """

    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter28.1 note

        Example
        ====
        ```python
        Chapter28_1().note()
        ```
        """
        print('chapter28.1 note as follow')
        print('28.1 矩阵的性质')
        print('矩阵运算在科学计算中非常重要')
        print('矩阵是数字的一个矩阵阵列,在python中使用np.matrix[[1,2],[3,4]]')
        print('矩阵和向量')
        print('单位矩阵')
        print('零矩阵')
        print('对角矩阵')
        print('三对角矩阵')
        print('上三角矩阵')
        print('下三角矩阵')
        print('置换矩阵')
        print('对称矩阵')
        print('矩阵乘法满足结合律,矩阵乘法对假发满足分配律')
        print('矩阵的F范数和2范数')
        print('向量的2范数')
        print('矩阵的逆,秩和行列式')
        print('定理28.1 一个方阵满秩当且仅当它为非奇异矩阵')
        print('定理28.2 当且仅当A无空向量,矩阵A列满秩')
        print('定理28.3 当且仅当A具有空向量时,方阵A是奇异的')
        print('定理28.4 (行列式的性质) 方阵A的行列式具有如下的性质')
        print('  如果A的任何行或者列的元素为0,则det(A)=0')
        print('  用常数l乘A的行列式任意一行(或任意一列)的各元素,等于用l乘A的行列式')
        print('  A的行列式的值与其转置矩阵A^T的行列式的值相等')
        print('  行列式的任意两行(或者两列)互换,则其值异号')
        print('定理28.5 当且仅当det(A)=0,一个n*n方阵A是奇异的')
        print('正定矩阵')
        print('定理28.6 对任意列满秩矩阵A,矩阵A\'A是正定的')
        print('练习28.1-1 证明：如果A和B是n*n对称矩阵,则A+B和A-B也是对称的')
        print('练习28.1-2 证明：(AB)\'=B\'A\',而且AA\'总是一个对称矩阵')
        print('练习28.1-3 证明：矩阵的逆是唯一的，即如果B和C都是A的逆矩阵,则B=C')
        print('练习28.1-4 证明：两个下三角矩阵的乘积仍然是一个下三角矩阵.',
            '证明：一个下三角(或者上三角矩阵)矩阵的行列式的值是其对角线上的元素之积',
            '证明：一个下三角矩阵如果存在逆矩阵,则逆矩阵也是一个下三角矩阵')
        print('练习28.1-5 证明：如果P是一个n*n置换矩阵,A是一个n*n矩阵,则可以把A的各行进行置换得到PA',
            '而把A的各列进行置换可得到AP。证明:两个置换矩阵的乘积仍然是一个置换矩阵',
            '证明：如果P是一个置换矩阵,则P是可逆矩阵,其逆矩阵是P^T,且P^T也是一个置换矩阵')
        print('练习28.1-6 设A和B是n*n矩阵，且有AB=I.证明:如果把A的第j行加到第i行而得到A‘',
            '则可以通过把B的第j列减去第i列而获得A’的逆矩阵B‘')
        print('练习28.1-7 设A是一个非奇异的n*n复数矩阵.证明：当且仅当A的每个元素都是实数时,',
            'A-1的每个元素都是实数')
        print('练习28.1-8 证明：如果A是一个n*n阶非奇异的对称矩阵,则A-1也是一个对称矩阵.',
            '证明：如果B是一个任意的m*n矩阵,则由乘积BAB^T给出的m*m矩阵是对称的')
        print('练习28.1-9 证明定理28.2。亦即，证明如果矩阵A为列满秩当且仅当若Ax=0,则说明x=0')
        print('练习28.1-10 证明:对任意两个相容矩阵A和B,rank(AB)<=min(rank(A),rank(B))',
            '其中等号仅当A或B是非奇异方阵时成立.(利用矩阵秩的另一种等价定义)')
        print('练习28.1-11 已知数x0,x1,...,xn-1,证明范德蒙德(Vandermonde)矩阵的行列式表达式')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_2:
    """
    chapter28.2 note and function
    """

    def __init__(self):
        pass

    def note(self):
        '''
        Summary
        ====
        Print chapter28.2 note

        Example
        ====
        ```python
        Chapter28_2().note()
        ```
        '''
        print('chapter28.2 note as follow')  
        print('28.2 矩阵乘法的Strassen算法')
        print('两个n*n矩阵乘积的著名的Strassen递归算法,其运行时间为Θ(n^lg7)=Θ(n^2.81)')
        print('对足够大的n,该算法在性能上超过了在25.1节中介绍的运行时间为Θ(n^3)的简易矩阵乘法算法MATRIX-MULTIPLY')
        print('算法概述')
        print('  Strassen算法可以看作是熟知的一种设计技巧--分治法的一种应用')
        print('  假设希望计算乘积C=AB,其中A、B和C都是n*n方阵.假定n是2的幂,把A、B和C都划分为四个n/2*n/2矩阵')
        print('  然后作分块矩阵乘法，可以得到递归式T(n)=8T(n/2)+Θ(n^2),但是T(n)=Θ(n^3)')
        print('Strassen发现了另外一种不同的递归方法,该方法只需要执行7次递归的n/2*n/2的矩阵乘法运算和Θ(n^2)次标量加法与减法运算')
        print('从而可以得到递归式T(n)=7T(n/2)+Θ(n^2),但是T(n)=Θ(n^2.81)')
        print('Strassen方法分为以下四个步骤')
        print(' 1) 把输入矩阵A和B划分为n/2*n/2的子矩阵')
        print(' 2) 运用Θ(n^2)次标量加法与减法运算,计算出14个n/2*n/2的矩阵A1,B1,A2,B2,...,A7,B7')
        print(' 3) 递归计算出7个矩阵的乘积Pi=AiBi,i=1,2,...,7')
        print(' 4) 仅使用Θ(n^2)次标量加法与减法运算,对Pi矩阵的各种组合进行求和或求差运算,',
            '从而获得结果矩阵C的四个子矩阵r,s,t,u')
        print('从实用的观点看,Strassen方法通常不是矩阵乘法所选择的方法')
        print('  1) 在Strassen算法的运行时间中,隐含的常数因子比简单的Θ(n^3)方法中的常数因子要大')
        print('  2) 当矩阵是稀疏的时候,为系数矩阵设计的方法更快')
        print('  3) Strassen算法不像简单方法那样具有数值稳定性')
        print('  4) 在递归层次中生成的子矩阵要消耗空间')
        # ! Strassen方法的关键就是对矩阵乘法作分治递归
        print('练习28.2-1 运用Strassen算法计算矩阵的乘积')
        print('矩阵的乘积为:')
        print(np.matrix([[1, 3], [5, 7]]) * np.matrix([[8, 4], [6, 2]]))
        print('练习28.2-2 如果n不是2的整数幂,应该如何修改Strassen算法,求出两个n*n矩阵的乘积',
            '证明修改后的算法的运行时间为Θ(n^lg7)')
        print('练习28.2-3 如果使用k次乘法(假定乘法不满足交换律)就能计算出两个3*3矩阵的乘积',
            '就能在o(n^lg7)时间内计算出两个n*n矩阵的乘积,满足上述条件的最大的k值是多少')
        print('练习28.2-4 V.Pan发现了一种使用132464次乘法的求68*68矩阵乘积的方法',
            '一种使用143640次乘法的求70*70矩阵乘积的方法',
            '一种使用155424次乘法的求72*72矩阵乘积的方法')
        print('练习28.2-5 用Strassen算法算法作为子程序,能在多长时间内计算出一个kn*n矩阵与一个n*kn矩阵的乘积')
        print('练习28.2-6 说明如何仅用三次实数乘法运算,就可以计复数a+bi与c+di的乘积.该算法应该把a,b,c和d作为输入,',
            '并分别生成实部ac-bd和虚部ad+bc的值') 
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_3:
    """
    chapter28.3 note and function
    """

    def __init__(self):
        pass

    def note(self):
        '''
        Summary
        ====
        Print chapter28.3 note

        Example
        ====
        ```python
        Chapter28_3().note()
        ```
        '''
        print('chapter28.3 note as follow')  
        print('28.3 求解线性方程组')
        print('对一组同时成立的线性方程组Ax=b求解时很多应用中都会出现的基本问题。一个线性系统可以表述为一个矩阵方程',
            '其中每个矩阵或者向量元素都属于一个域,如果实数域R')
        print('LUP分解求解线性方程组')
        print('LUP分解的思想就是找出三个n*n矩阵L,U和P,满足PA=LU')
        print('  其中L是一个单位下三角矩阵,U是一个上三角矩阵,P是一个置换矩阵')
        print('每一个非奇异矩阵A都有这样一种分解')
        print('对矩阵A进行LUP分解的优点是当相应矩阵为三角矩阵(如矩阵L和U),更容易求解线性系统')
        print('在计算出A的LUP分解后,就可以用如下方式对三角线性系统进行求解,也就获得了Ax=b的解')
        print('对Ax=b的两边同时乘以P,就得到等价的方程组PAx=Pb,得到LUx=Pb')
        print('正向替换与逆向替换')
        print('  如果已知L,P和b,用正向替换可以在Θ(n^2)的时间内求解下三角线性系统',
            '用一个数组pi[1..n]来表示置换P')
        print('LU分解的计算')
        print('  把执行LU分解的过程称为高斯消元法.先从其他方程中减去第一个方程的倍数',
            '以便把那些方程中的第一个变量消去')
        print('  继续上述过程,直至系统变为一个上三角矩阵形式,这个矩阵都是U.矩阵L是由使得变量被消去的行的乘数所组成')
        print('LUP分解的计算')
        print('  一般情况下,为了求线性方程组Ax=b的解,必须在A的非对角线元素中选主元以避免除数为0',
            '除数不仅不能为0,也不能很小(即使A是非奇异的),否则就会在计算中导致数值不稳定.因此,所选的主元必须是一个较大的值')
        print('  LUP分解的数学基础与LU分解相似。已知一个n*n非奇异矩阵A,并希望计算出一个置换矩阵P,一个单位下三角矩阵L和一个上三角矩阵U,并满足条件PA=LU')
        print('练习28.3-1 运用正向替换法求解下列方程组')
        print('练习28.3-2 求出下列矩阵的LU分解')
        print('练习28.3-3 运用LUP分解来求解下列方程组')
        print('练习28.3-4 试描述一个对角矩阵的LUP分解')
        print('练习28.3-5 试描述一个置换矩阵A的LUP分解,并证明它是唯一的')
        print('练习28.3-6 证明：对所有n>=1,存在具有LU分解的奇异的n*n矩阵')
        print('练习28.3-7 在LU-DECOMPOSITION中,当k=n时是否有必要执行最外层的for循环迭代?',
            '在LUP-DECOMPOSITION中的情况又是怎样?')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_4:
    """
    chapter28.4 note and function
    """
    def note(self):
        """
        Summary
        ====
        Print chapter28.4 note

        Example
        ====
        ```python
        Chapter28_4().note()
        ```
        """
        print('chapter28.4 note as follow')  
        print('28.4 矩阵求逆')
        print('在实际应用中,一般并不使用逆矩阵来求解线性方程组的解,而是运用一些更具数值稳定性的技术,如LUP分解求解线性方程组')
        print('但是,有时仍然需要计算一个矩阵的逆矩阵.可以利用LUP分解来计算逆矩阵')
        print('此外,还将证明矩阵乘法和计算逆矩阵问题是具有相同难度的两个问题,即(在技术条件限制下)可以使用一个算法在相同渐进时间内解决另外一个问题')
        print('可以使用Strassen矩阵乘法算法来求一个矩阵的逆')
        print('确实,正是由于要证明可以用比通常的办法更快的算法来求解线性方程组,才推动了最初的Strassen算法的产生')
        print('根据LUP分解计算逆矩阵')
        print('  假设有一个矩阵A的LUP分解,包括三个矩阵L,U,P,并满足PA=LU')
        print('  如果运用LU-SOLVE,则可以在Θ(n^2)的运行时间内,求出形如Ax=b的线性系统的解')
        print('  由于LUP分解仅取决于A而不取决于b,所以就能够再用Θ(n^2)的运行时间,求出形如Ax=b\'的另一个线性方程组的解')
        print('  一般地,一旦得到了A的LUP分解,就可以在Θ(kn^2)的运行时间内,求出k个形如Ax=b的线性方程组的解,这k个方程组只有b不相同')
        print('矩阵乘法与逆矩阵')
        print('  对矩阵乘法可以获得理论上的加速,可以相应地加速求逆矩阵的运算')
        print('  从下面的意义上说,求逆矩阵运算等价于矩阵乘法运算',
            '如果M(n)表示求两个n*n矩阵乘积所需要的时间,则有在O(M(n))时间内对一个n*n矩阵求逆的方法',
            '如果I(n)表示对一个非奇异的n*n矩阵求逆所需的时间,则有在O(I(n))时间内对两个n*n矩阵相乘的方法')
        print('定理28.7 (矩阵乘法不比求逆矩阵困难) 如果能在I(n)时间内求出一个n*n矩阵的逆矩阵',
            '其中I(n)=Ω(n^2)且满足正则条件I(3n)=O(I(n))时间内求出两个n*n矩阵的乘积')
        print('定理28.8 (求逆矩阵运算并不比矩阵乘法运算更困难) 如果能在M(n)的时间内计算出两个n*n实矩阵的乘积',
            '其中M(n)=Ω(n^2)且M(n)满足两个正则条件：对任意的0<=k<=n有M(n+k)=O(M(n)),以及对某个常数c<1/2有M(n/2)<=cM(n)',
            '则可以在O(M(n))时间内求出任何一个n*n非奇异实矩阵的逆矩阵')
        print('练习28.4-1 设M(n)是求n*n矩阵的乘积所需的时间,S(n)表示求n*n矩阵的平方所需时间',
            '证明:求矩阵乘积运算与求矩阵平方运算实质上难度相同：一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的矩阵平方算法,',
            '一个S(n)时间的矩阵平方算法蕴含着一个O(S(n))时间的矩阵相乘算法')
        print('练习28.4-2 设M(n)是求n*n矩阵乘积所需的时间,L(n)为计算一个n*n矩阵的LUP分解所需要的时间',
            '证明：求矩阵乘积运算与计算矩阵LUP分解实质上难度相同:一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的矩阵LUP分解算法',
            '一个L(n)时间的矩阵LUP分解算法蕴含着一个O(L(n))时间的矩阵相乘算法')
        print('练习28.4-3 设M(n)是求n*n矩阵的乘积所需的时间,D(n)表示求n*n矩阵的行列式的值所需要的时间',
            '证明：求矩阵乘积运算与求行列式的值实质上难度相同：一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的行列式算法',
            '一个D(n)时间的行列式算法蕴含着一个O(D(n)时间的矩阵相乘算法')
        print('练习28.4-4 设M(n)是求n*n布尔矩阵的乘积所需的时间,T(n)为找出n*n布尔矩阵的传递闭包所需要的时间',
            '证明：一个M(n)时间的布尔矩阵相乘算法蕴含着一个O(M(n)lgn)时间的传递闭包算法,一个T(n)时间的传递闭包算法蕴含着一个O(T(n))时间的布尔矩阵相乘算法')
        print('练习28.4-5 当矩阵元素属于整数模2所构成的域时,基于定理28.8的求逆矩阵算法的是否能够运行？')
        print('练习28.4-6 推广基于定理28.8的求逆矩阵算法,使之能处理复矩阵的情形,并证明所给出的推广方法是正确的')
        print('   提示：用A的共轭转置矩阵A*来代替A的转置矩阵A^T,把A^T中的每个元素用其共轭复数代替就得到A*,也就是Hermitian转置')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_5:
    """
    chapter28.5 note and function
    """
    def note(self):
        """
        Summary
        ====
        Print chapter28.5 note

        Example
        ====
        ```python
        Chapter28_5().note()
        ```
        """
        print('chapter28.5 note as follow') 
        print('28.5 对称正定矩阵与最小二乘逼近') 
        print('对称正定矩阵有许多有趣而很理想的性质。例如，它们都是非奇异矩阵,并且可以对其进行LU分解而无需担心出现除数为0的情况')
        print('引理28.9 任意对称矩阵都是非奇异矩阵')
        print('引理28.10 如果A是一个对称正定矩阵,则A的每一个主子式都是对称正定的')
        print('设A是一个对称正定矩阵,Ak是A的k*k主子式,矩阵A关于Ak的Schur补定义为S=C-BAk^-1B^T')
        print('引理28.11 (Schur补定理) 如果A是一个对称正定矩阵,Ak是A的k*k主子式.则A关于Ak的Schur补也是对称正定的')
        print('推论28.12 对一个对称正定矩阵进行LU分解不会出现除数为0的情形')
        print('最小二乘逼近')
        print('对给定一组数据的点进行曲线拟合是对称正定矩阵的一个重要应用,假定给定m个数据点(x1,y1),(x2,y2),...,(xm,ym)',
            '其中已知yi受到测量误差的影响。希望找出一个函数F(x),满足对i=1,2,...,m,有yi=F(xi)+qi')
        print('其中近似误差qi是很小的,函数F(x)的形式依赖于所遇到的问题,在此,假定它的形式为线性加权和F(x)=∑cifi(x)')
        print('其中和项的个数和特定的基函数fi取决于对问题的了解,一种选择是fi(x)=x^j-1,这说明F(x)是一个x的n-1次多项式')
        print('这样一个高次函数F尽管容易处理数据,但也容易对数据产生干扰,并且一般在对未预见到的x预测其相应的y值时,其精确性也是很差的')
        print('为了使逼近误差最小,选定使误差向量q的范数最小,就得到一个最小二乘解')
        print('统计学中正态方程A^TAc=A^Ty')
        print('伪逆矩阵A+=(A^TA)^-1A^T')
        print('练习28.5-1 证明：对称正定矩阵的对角线上每一个元素都是正值')
        print('练习28.5-2 设A=[[a,b],[b,c]]是一个2*2对称正定矩阵,证明其行列式的值ac-b^2是正的')
        print('练习28.5-3 证明：一个对称正定矩阵中值最大的元素处于其对角线上')
        print('练习28.5-4 证明：一个对称正定矩阵的每一个主子式的行列式的值都是正的')
        print('练习28.5-5 设Ak表示对称正定矩阵A的第k个主子式。证明在LU分解中,det(Ak)/det(Ak-1)是第k个主元,为方便起见,设det(A0)=1')
        print('练习28.5-6 最小二乘法求')
        print('练习28.5-7 证明：伪逆矩阵A+满足下列四个等式：')
        print('  AA^+A=A')
        print('  A^+AA^+=A^+')
        print('  (AA^+)^T=AA^+')
        print('  (A^+A)^T=A^+A')
        print('思考题28-1 三对角线性方程组')
        print(' 1) 证明：对任意的n*n对称正定的三对角矩阵和任意n维向量b,通过进行LU分解可以在O(n)的时间内求出方程Ax=b的解',
            '论证在最坏情况下,从渐进意义上看,基于求出A^-1的任何方法都要花费更多的时间')
        print(' 2) 证明：对任意的n*n对称正定的三对角矩阵和任意n维向量b,通过进行LUP分解,'<
            '可以在O(n)的时间内求出方程Ax=b的解')
        print('思考题28-2 三次样条插值')
        print('  将一个曲线拟合为n个三次多项式组成')
        print('  用自然三次样条可以在O(n)时间内对一组n+1个点-值对进行插值')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

chapter28_1 = Chapter28_1()
chapter28_2 = Chapter28_2()
chapter28_3 = Chapter28_3()
chapter28_4 = Chapter28_4()
chapter28_5 = Chapter28_5()

def printchapter28note():
    """
    print chapter28 note.
    """
    print('Run main : single chapter twenty-eight!')
    chapter28_1.note()
    chapter28_2.note()
    chapter28_3.note()
    chapter28_4.note()
    chapter28_5.note()

# python src/chapter28/chapter28note.py
# python3 src/chapter28/chapter28note.py

if __name__ == '__main__':  
    printchapter28note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter29/chapter29note.py
# python3 src/chapter29/chapter29note.py
"""

Class Chapter29_1

Class Chapter29_2

Class Chapter29_3

Class Chapter29_4

Class Chapter29_5

"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np

class Chapter29_1:
    """
    chapter29.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter29.1 note

        Example
        ====
        ```python
        Chapter29_1().note()
        ```
        """
        print('chapter29.1 note as follow')
        print('第29章 线性规划')
        print('在给定有限的资源和竞争约束情况下，很多问题都可以表述为最大化或最小化某个目标')
        print('如果可以把目标指定为某些变量的一个线性函数,而且如果可以将资源的约束指定为这些变量的等式或不等式',
            '则得到一个线性规划问题.线性规划出现在许多世纪应用中')
        print('比如如下线性规划问题')
        print('argmin(x1+x2+x3+x4)')
        print('满足约束条件')
        print('-2 * x1 + 8 * x2 +  0 * x3 + 10 * x4 >= 50')
        print(' 5 * x1 + 2 * x2 +  0 * x3 +  0 * x4 >= 100')
        print(' 3 * x1 - 5 * x2 + 10 * x3 -  2 * x4 >= 25')
        print(' x1 >= 0; x2 >= 0; x3 >= 0; x4 >= 0')
        print('一般线性规划')
        print('  在一般线性规划的问题中,希望最优化一个满足一组线性不等式约束的线性函数。',
            '已知一组实数a1,a2,...,an和一组变量x1,x2,...,xn,在这些变量上的一个线性函数f定义为：',
            'f(x1,x2,...,xn)=a1x1+a2x2+...+anxn')
        print('  如果b是一个实数而f是一个线性函数,则等式f(x1,x2,...,xn)=b是一个线性等式')
        print('  不等式f(x1,x2,...,xn)<=b和f(x1,x2,...,xn)>=b都是线性不等式')
        print('用线性约束来表示线性等式或线性不等式')
        print('在线性规划中,不允许严格的不等式')
        print('正式地说,线性规划问题是这样的一种问题,要最小化或最大化一个受限一组有限的线性约束的线性函数')
        print('如果是要最小化,则称此线性规划为最小化线性规划;如果是要最大化,则称此线性规划为最大化线性规划')
        print('虽然有一些线性规划的多项式时间算法。但是单纯形法是最古老的线性规划算法.',
            '单纯形算法在最坏的情况下不是在多项式时间内运行,但是相当有效,而且在实际中被广泛使用')
        print('比如双变量的线性规划直接在笛卡尔直角坐标系中表示出可行域和目标函数曲线即可')
        print('线性规划概述')
        print('  非正式地,在标准型中的线性规划是约束为线性不等式的线性函数的最大化',
            '而松弛型的线性规划是约束为线性等式的线性函数的最大化')
        print('  通常使用标准型来表示线性规划,但当描述单纯形算法的细节时,使用松弛形式会比较方便')
        print('受m个线性不等式约束的n个变量上的线性函数的最大化')
        print('如果有n个变量,每个约束定义了n维空间中的一个半空间.这些半空间的交集形成的可行区域称作单纯形')
        print('目标函数现在成为一个超平面,而且因为它的凸性,故仍然有一个最优解在单纯形的一个顶点上取得的')
        print('单纯形算法以一个线性规划作为输入,输出它的一个最优解.从单纯形的某个顶点开始,执行一系列的迭代',
            '在每次迭代中,它沿着单纯形的一条边从当前定点移动到一个目标值不小于(通常是大于)当前顶点的相邻顶点',
            '当达到一个局部的最大值,即一个顶点的目标值大于其所有相邻顶点的目标值时,单纯形算法终止.')
        print('因为可行区域是凸的而且目标函数是线性的,所以局部最优事实上是全局最优的')
        print('将使用一个称作\"对偶性\"的概念来说明单纯形法算法输出的解的确是最优的')
        print('虽然几何观察给出了单纯形算法操作过程的一个很好的直观观察',
            '但是在讨论单纯形算法的细节时,并不显式地引用它.相反地，采用一种代数方法,首先将已知的线性规划写成松弛型,即线性等式的集合',
            '这些线性等式将表示某些变量,称作\"基本变量\",而其他变量称作\"非基本变量\".从一个顶点移动到另一个顶点伴随着将一个基本变量',
            '变为非基本变量,以及将一个非基本变量变为基本变量.',
            '这个操作称作一个\"主元\",而且从代数的观点来看,只不过是将线性规划重写成等价的松弛型而已')
        print('识别无解的线性规划,没有有限最优解的线性规划,以及原点不是可行解的线性规划 ')
        print('线性规划的应用')
        print('  线性规划有大量的应用。任何一本运筹学的教科书上都充满了线性规划的例子')
        print('  线性规划在建模和求解图和组合问题时也很有用,可以将一些图和网络流问题形式化为线性规划')
        print('  还可以利用线性规划作为工具，来找出另一个图问题的近似解')
        print('线性规划算法')
        print('  当单纯形法被精心实现时,在实际中通常能够快速地解决一般的线性规划',
            '然而对于某些刻意仔细设计的输入，单纯形法可能需要指数时间')
        print('  线性规划的第一个多项式时间算法是椭圆算法,在实际中运行缓慢')
        print('  第二类指数时间的算法称为内点法,与单纯形算法(即沿着可行区域的外部移动,并在每次迭代中维护一个为单纯形的顶点的可行解)相比',
            '这些算法在可行区域的内部移动.中间解尽管是可行的,但未必是单纯形的顶点,但最终的解是一个顶点')
        print('  对于大型输入,内点法的性能可与单纯形算法相媲美,有时甚至更快')
        print('  仅找出整数线性规划这个问题的一个可行解就是NP-难度的;因为还没有已知的多项式时间的算法能解NP-难度问题')
        print('  所以没有已知的整数线性规划的多项式算法.相反地,一般的线性规划问题可以在多项式时间内求解')
        print('  定义线性规划其变量为x=(x1,x2,...,xn),希望引用这些变量的一个特定设定,将使用记号x`=(x1`,x2`,...,xn`)')
        print('29.1 标准型和松弛型')
        print('  在标准型中的所有约束都是不等式,而在松弛型中的约束都是等式')
        print('标准型')
        print('  已知n个实数c1,c2,...,cn;m个实数b1,b2,...,bm;以及mn个实数aij,其中i=1,2,...,m,而j=1,2,...,n',
            '希望找出n个实数x1,x2,...,xn来最大化目标函数∑cjxj,满足约束∑aijxj<=bi,i=1,2,...,m;xj>=0',
            'n+m个不等式约束,n个非负性约束')
        print('  一个任意的线性规划需要有非负性约束,但是标准型需要,有时将一个线性规划表示成一个更紧凑的形式会比较方便')
        print('  如果构造一个m*n矩阵A=(aij),一个m维的向量b=(bi),一个n维的向量c=(cj),以及一个n维的向量x=(xj)',
            '最大化c^Tx,满足约束Ax<=b,x>=0')
        print('  c^Tx是两个向量的内积,Ax是一个矩阵向量乘积,x>=0表示向量x的每个元素都必须是非负的')
        print('  称满足所有约束的变量x`的设定为可行解,而不满足至少一个约束的变量x`的设定为不可行解')
        print('  称一个解x`拥有目标值c^T.在所有可行解中其目标值最大的一个可行解x`是一个最优解,且称其目标值c^Tx`为最优目标值')
        print('  如果一个线性规划没有可行解,则称此线性规划不可行;否则它是可行的')
        print('  如果一个线性规划有一些可行解但没有有限的最优目标值,则称此线性规划是无界的')
        print('将线性规划转换为标准型')
        print('  已知一个最小化或最大化的线性函数受若干线性约束,总可以将这个线性规划转换为标准型')
        print('  一个线性规划可能由于如下4个原因而不是标准型')
        print('    (1) 目标函数可能是一个最小化,而不是最大化')
        print('    (2) 可能有的变量不具有非负性约束')
        print('    (3) 可能有等式约束，即有一个等号而不是小于等于号')
        print('    (4) 可能有不等式约束,但不是小于等于号,而是一个大于等于号')
        print('当把一个线性规划L转化为另一个线性规划L\'时,希望有性质：从L\'的最优解能得到L的最优解.为解释这个思想,',
            '说两个最大化线性规划L和L\'是等价的')
        print('将一个最小化线性规划L转换成一个等价的最大化线性规划L\',简单地对目标函数中的系数取负值即可')
        print('因为当且仅当x>=y和x<=y时x=y,所以可以将线性规划中的等式约束用一对不等式约束来替代')
        print('在每个等式约束上重复这个替换，就得到全是不等式约束的线性规划')
        print('将线性规划转换为松弛型')
        print('  为了利用单纯形算法高效地求解线性规划,通常将它表示成其中某些约束是等式的形式')
        print('  ∑aijxj <= bi是一个不等式约束,引入一个新的松弛变量s,重写不等式约束')
        print('  s = bi - ∑aijxj; s >= 0')
        print('  s度量了等式左边和右边之间的松弛或差别.因为当且仅当等式和不等式都为真时不等式为真')
        print('  所以可以对线性规划的每个不等式约束应用这个转换,得到一个等价的线性规划,其中只有不等式是非负约束')
        print('  当从标准型转换到松弛型时,将使用xn+i(而不是s)来表示与第i个不等式关联的松弛变量')
        print('  因此第i个约束是xn+i = bi - ∑aijxj 以及非负约束xn+i >= 0')
        print('练习29.1-1 线性规划表示简洁记号形式,n,m,A,b分别是什么')
        print('练习29.1-2 给出题目线性规划的3个可行解,每个解的目标值是多少')
        print('练习29.1-3 线性规划转换为松弛型后,N、B、A、b、c和v分别是什么')
        print('练习29.1-4 线性规划转换为标准型')
        print('练习29.1-5 线性规划转换为松弛型')
        print('练习29.1-6 说明下列线性规划不可行')
        print('练习29.1-7 说明下列线性规划是无界的')
        print('练习29.1-8 假设有一个n个变量和m个约束的线性规划,且假设将其转换为成标准型',
            '请给出所得线性规划中变量和约束个数的一个上界')
        print('练习29.1-9 请给出一个线性规划的例子,其中可行区域是无界的,但最优解的值是有界的')
        # python src/chapter29/chapter29note.py
        # python3 src/chapter29/chapter29note.py

class Chapter29_2:
    """
    chapter29.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter29.2 note

        Example
        ====
        ```python
        Chapter29_2().note()
        ```
        """
        print('chapter29.2 note as follow')
        print('29.2 将问题表达为线性规划')
        print('虽然本章的重点在单纯形算法上,但是识别出一个问题是否可以形式化为一个线性规划是很重要的')
        print('一旦一个问题被形式化成一个多项式规模的线性规划,它可以用椭圆法或内点法在多项式时间内解决')
        print('一些线性规划的软件包可以高效地解决问题')
        print('线性规划问题的实际例子：单源最短路径问题，最大流问题，最小费用流问题')
        print('最小费用流问题有一个不是基于线性规划的多项式时间算法')
        print('多商品流问题：它的唯一已知的多项式算法是基于线性规划的')
        print('最短路径')
        print('  在单对最短路径问题中，已知有一个带权有向图G=(V,E),',
            '加权函数w:E->R将边映射到实数值的权值,一个源顶点s,一个目的顶点t')
        print('  希望计算从s到t的一条最短路径的权值d[t],为把这个问题表示成线性规划',
            '需要确定变量和约束的集合来定义何时有从s到t的一条最短路径')
        print('  Bellman-Ford算法做的就是这个.当Bellman-Ford算法中止时,对每个顶点v,计算了一个值d[v],',
            '使得对每条边(u,v)∈E,有d[v]<=d[u]+w(u,v).源顶点初始得到一个值d[s]=0,以后也不会改变',
            '因此得到如下的线性规划,来计算从s到t的最短路径的权值,最小化d[t]',
            '满足约束d[v]<=d[u]+w(u,v),对每条边(u,v)∈E,d[s]=0',
            '在这个线性规划中,有|V|个变量d[v],每个顶点v∈V各有一个.有|E|+1个约束',
            '每条边各有一个再加上源顶点总是有值0的额外约束')
        print('最大流')
        print('  最大流问题也可以表示成线性规划,已知一个有向图G=(V,E),其中每条边(u,v)∈E有一个非负的容量c(u,v)>=0',
            '以及两个特别的顶点:源s和汇t.流是一个实数值的函数f:V*V->R,满足三个性质：容量限制,斜对称性,流守恒性')
        print('  最大流是满足这些约束和最大化流量值的流,其中流量值是从源流出的总流量。因此,流满足线性约束,且流的值是一个线性函数',
            '还假设了如果(u,v)∉E,则c(u,v)=0,可最大化∑f(s,v)')
        print('  满足约束f(u,v)<=c(u,v),对每个u,v∈V')
        print('  满足约束f(u,v)=-f(v,u),对每个u,v∈V')
        print('  ∑f(u, v)=0,对每个u∈V-{s,t}')
        print('这个线性规划有|V|^2个变量,对应于每一顶点之间的流,且有2|V|^2+|V|-2个约束')
        print('通常求解一个较小规模的线性规划更加有效。线性规划有一个流和每对(u, v)∉E的顶点u,v的容量为0',
            '把这个线性规划重写成有O(V+E)个约束的形式会更有效')
        print('最小费用流')
        print('  事实上,为一个问题特别设计一个有效的算法,如用于单源最短路径的Dijkstra算法,或者最大流的push-relabel方法',
            '经常在理论和实践中都比线性规划更加有效')
        print('  线性规划的真正力量来自其求解新问题的能力')
        print('  考虑最大流问题的如下一般化.假设每条边(u,v)除了有一个容量c(u,v)外,还有一个实数值的费用a(u,v),通过边(u,v)传送f(u,v)个单位的流',
            '那么发生了一个费用a(u,v)f(u,v).同时还给定了一个流目标d,希望从s发送单个单位的流到t',
            '使得流上发生的总费用∑a(u,v)f(u,v)最小.这个问题被称为最小费用流问题')
        print('  有特别为最小费用流设计的多项式时间算法,然而可以将最小费用流问题表示成一个线性规划',
            '这个线性规划看上去和最大流问题相似,有流量为准确的d个单位的额外约束,以及最小化费用的新的目标函数')
        print('多商品流')
        print('  仍然给定一个有向图G=(V,E),其中每条边(u,v)∈E有一个非负的容量c(u,v)>=0.如同在最大流问题中一样,',
            '隐含地假设对于(u,v)∉E有c(u,v)=0,另外,还已知k中不同的商品,K1,K2,...,Kk',
            '其中商品i用元组Ki=(si,ti,di)来指定。这里,si是商品i的源;ti是商品i的汇;di是需求',
            '即商品i从si到ti所需的流量值.将商品i的流用fi表示(因此fi(u,v)是商品i从顶点u到顶点v的流),定义为一个满足流量守恒、斜对称性和容量约束的实数值函数',
            '现在定义汇聚流f(u,v)为各种商品流的总和,因此f(u,v)=∑fi(u,v)')
        print('在边(u,v)上的汇聚流不能超过边(u,v)的容量。这个约束包含了每个商品的容量约束。以商鞅的方式来描述这个问题,没有东西要最小化',
            '只需要确定是否能找到这样的一个流。因此，用一个“空”的目标函数来写这个线性规划，满足约束')
        print('∑fi(u,v)<=c(u,v),对每个u,v∈V')
        print('fi(u,v)=-fi(v,u),对每个i=1,2,...,k;并且对每个u,v∈V')
        print('∑fi(u,v)=0,对每个i=1,2,...,k;并且对每个u∈V-{si,ti}')
        print('∑fi(s,v)=di,对每个i=1,2,...,k')
        print('这个问题唯一已知的多项式时间算法是将它表示成一个线性规划,然后用一个多项式时间线性规划来解决')
        print('练习29.2-1 将单对最短路径线性规划从上述公式转换成标准型')
        print('练习29.2-2 寻找从结点s到结点y的最短路径相对应的线性规划')
        print('练习29.2-3 在单源最短路径问题中,想要找出从源点s到所有顶点v∈V的最短路径权值',
            '给定一个图G,写出一个线性规划,它的解具有性质:对每个顶点v∈V,d[v]是从s到v的最短路径的权值')
        print('练习29.2-4 详细写出在图中寻找最大流所对应的线性规划')
        print('练习29.2-5 重写最大流式的线性规划,使它只使用O(V+E)个约束')
        print('练习29.2-6 写出一个线性规划,已知一个二分图G,求解最大二分匹配问题')
        print('练习29.2-7 在最小费用多商品流问题中,给定有向图G=(V,E),其中每条边(u,v)∈E有一个非负的容量c(u,v)>=0',
            '以及一个费用a(u,v).如同在多商品流问题中一样,已知k种不同的商品,K1,K2,...,Kk,其中商品i用元组Ki=(si,ti,di)来指定',
            '如同在多商品流问题中一样,为商品i定义流fi,在边(u,v)上定义汇聚流f(u,v).',
            '一个可行流是在每条边(u,v)上的汇聚流不超过边(u,v)的容量',
            '一个流的费用是∑a(u,v)f(u,v),目标是寻找最小费用的可行流。将这个问题表示为一个线性规划')
        # python src/chapter29/chapter29note.py
        # python3 src/chapter29/chapter29note.py

class Chapter29_3:
    """
    chapter29.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter29.3 note

        Example
        ====
        ```python
        Chapter29_3().note()
        ```
        """
        print('chapter29.3 note as follow')
        print('29.3 单纯形算法')
        print('单纯形算法是求解线性规划的古典方法。它的执行时间在最坏的情况下并不是多项式.',
            '然而确实加深了对线性规划的理解,并且在实际中通常相当快速')
        print('除了几何解释外,单纯形算法与高斯消元法有些类似的地方,高斯消元法从解未知的一个线性等式系统开始')
        print('在每次迭代中,将这个系统重写为具有一些额外结构的等价形式',
            '经过一定次数的迭代后,已经重写这个系统,使得它的解很容易得到,',
            '单纯形算法以一个相似的方式进行,而且可以将其看作是在不等式上的高斯消元法')
        print('现在描述在单纯形算法迭代背后的主要思想.和每次迭代关联的基本解可以很容易地从线性规划的松弛型中得到：',
            '将每个非基本变量设为0,并从等式约束中计算基本变量的值')
        print('一个基本解总是对应于单纯形的一个顶点,在代数上,一次迭代将一个松弛型转换成一个等价的松弛型')
        print('相应的基本可行解的目标值不小于前一次迭代中的目标值(通常是大于).为了达到目标值的这种递增',
            '选择一个非基本变量,使得如果是要从0开始增加这个变量的值,则目标值也会增加')
        print('可以在变量上增加的数值受其他约束限制。特别是，要增加它直到某个基本变量变为0为止',
            '然后重写松弛型,将这个基本变量和所选的非基本变量的角色互换,虽然使用了变量的一个特殊设定来指导这个算法',
            '而且还将在证明中使用它,但这个算法并没有明显地维护这个解,它只是重写线性规划直到最优解变得“明显”为止')
        print('单纯形法的一个例子')
        print('  为了利用单纯形算法,必须将线性规划转换成松弛型;除了是一个代数操作外,松弛也是一个有用的算法概念',
            '每个变量有一个对应的非负约束;称一个等式约束对于它的非基本变量的一个特殊设定是紧(tight)的',
            '(如果它们导致这个约束的基本变量为0的话).类似地,导致一个基本变量变为负值的非基本变量的设定违反了这个约束',
            '所以,松弛变量显式地维护每个约束距离紧的程度,帮助我们确定可以增加多少非基本变量的值而不违反任何约束')
        print('  单纯形算法的每次迭代会重写等式集合和目标函数,来将一个不同的变量集合放在右边.因此,重写过的问题会有一个不同的基本解',
            '强调重写不会改变基本的线性规划;在一次迭代中的问题与前一次迭代中的问题有着相同的可行解集合',
            '然而,问题确实会与前一次迭代的问题有着不同的基本解')
        print('  如果一个基本解也是可行的,则称其为基本可行解(basic feasible solution),在单纯形算法的执行过程中,基本解几乎总是基本可行解',
            '在单纯形算法的前一次迭代中,基本解可能不是可行的')
        print('  在每次迭代中,目标是重新表达线性规划,来让基本解有一个更大的目标值.选择一个在目标函数中系数的为正值的非基本变量xe,',
            '而且尽可能增加xe的数值而不违反任何约束。变量xe成为基本变量,某个其他变量xl成为非基本变量.其他基本变量和目标函数的值都可能改变')
        print('正式的单纯形算法')
        print('  现在可以对单纯形算法进行形式化了；进一步讨论的问题：')
        print('   (1) 如何确定一个线性规划是不是可行的')
        print('   (2) 如果线性规划是可行的,但初始基本解是不可行的,该做什么')
        print('   (3) 应该如何确定一个线性规划是无界的')
        print('   (4) 应该如何选择换入变量和换出变量')
        print('引理29.3 令I表示一个下标的集合.对每个i∈I,令ai和bi表示实数,令xi表示一个实数值的变量,令y表示任意的实数',
            '假设对于xi的任何取值,有∑aixi=y+∑bixi')
        print('引理29.4 令(A,b,c)表示一个线性规划的标准型。已知基本变量的集合B。则所属的松弛型被唯一确定')
        print('中止性')
        print('  单纯形算法的每次迭代会增加和基本解相关联的目标值.SIMPLEX的任何迭代都不会减小和基本解相关联的目标值',
            '遗憾的是,可能会有一次迭代维持目标值不变.这个现象叫做退化,开始详细地研究它')
        print('  退化是唯一可能让单纯形算法不终止的方式')
        print('引理29.5 如果SINPLEX在至多(n+m,m)次迭代内不能中止,则它是循环的')
        print('引理29.6 如果在SIMPLEX的第3行和第8行,总是选择具有最小下标的变量来打破一样的目标值,那么SIMPLEX必定会终止')
        print('引理29.7 假设INITIALIZE-SIMPLEX返回一个基本解可行的松弛型,则SIMPLEX要么报告线性规划是无界的,',
            '要么它在至多(n+m,m)次迭代内得到一个可行解来终止')
        print('练习29.3-1 略')
        print('练习29.3-2 略')
        print('练习29.3-3 假设将一个标准型的线性规划(A,b,c转换为松弛型),说明当且仅当bi>=0时(i=1,2,...,m),基解是可行的')
        print('练习29.3-4 使用SIMPLEX求解下面的线性规划')
        print('练习29.3-5 使用SIMPLEX求解下面的线性规划')
        print('练习29.3-6 使用SIMPLEX求解下面的线性规划')
        # python src/chapter29/chapter29note.py
        # python3 src/chapter29/chapter29note.py

class Chapter29_4:
    """
    chapter29.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter29.4 note

        Example
        ====
        ```python
        Chapter29_4().note()
        ```
        """
        print('chapter29.4 note as follow')
        print('29.4 对偶性')
        print('已经证明了在某些假设下,SIMPLEX是会终止的.然而,还没有说明它确实能找到线性规划的一个最优解',
            '为此引入一个有效的概念,叫做线性规划对偶性')
        print('对偶性是个非常重要的性质.在一个最优化问题中,一个对偶问题的识别几乎总是伴随着一个多项式时间算法的发现',
            '对偶性也可以用来证明某个解的确是最优解')
        print('假设已知一个最大流问题的实例,要寻找的是一个值为|f|的流f.根据最大流最小割定理,如果可以找到一个割的值也是|f|',
            '就证实了f确实是一个最大流.这是对偶性的一个例子：已知一个最大化问题,定义一个相关的最小化问题,来让这两个问题有相同的最优目标值')
        print('已知一个目标是最大化的线性规划,要描述如何制定一个对偶线性规划,它的目标是最小化,而且最优值与原始线性规划的相同')
        print('在表示对偶线性规划时,称原始的线性规划为原')
        print('为了构造对偶问题,将最大化改成最小化,将约束右边的与目标函数的系数的角色互换,并且将小于等于号改成大于等于号',
            '在原问题的m个约束中,每一个在对偶问题中都有一个对应的变量yi;在对偶问题的n个约束中,每一个在原问题中都有一个对应的变量xj')
        print('推论29.9 令x`表示原线性规划(A,b,c)的一个可行解,且令y`表示相应的对偶问题的一个可行解',
            '如果∑cjx`j=∑biy`i,则x`和y`分别是原线性规划和对偶线性规划的最优解')
        print('定理29.10(线性规划对偶性) 假设SIMPLEX在原线性规划(A,b,c)上返回值x`=(x`1,x`2,x`3,...,x`n)',
            '令N和B表示最终松弛型的非基本变量和基本变量的集合,令c`表示最终松弛型中的系数',
            '令y`=(y`1,y`2,y`3,...,y`m),则x`是原线性规划的一个最优解,y`是对偶线性规划的一个最优解,而且∑cjx`j=∑biy`i')
        print('练习29.4-1 写出某线性规划的对偶问题')
        print('练习29.4-2 假设有一个不是标准型的线性规划.可以通过先将其转换成标准型,再取对偶来产生它的对偶问题',
            '然而,更方便的是能够直接产生对偶问题。说明已知一个任意的线性规划,如何直接取该线性规划的对偶')
        print('练习29.4-3 写出行中给出的最大流线性规划的对偶问题。说明如何将这个形式理解成一个最小割问题')
        print('练习29.4-4 写出行中给出的最小费用流线性规划的对偶问题.说明如何将这个问题用图和流来解释')
        print('练习29.4-5 证明：线性规划的对偶是原线性规划')
        print('练习29.4-6 哪一个结果可以解释成最大流的弱对偶')
        # python src/chapter29/chapter29note.py
        # python3 src/chapter29/chapter29note.py

class Chapter29_5:
    """
    chapter29.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter29.5 note

        Example
        ====
        ```python
        Chapter29_5().note()
        ```
        """
        print('chapter29.5 note as follow')
        print('29.5 初始基本可行解')
        print('找出一个初始解')
        print('  线性规划是否有可行解,如果有,则给出一个基本可行的松弛型,一个线性规划可能是可行的,但是初始基本解可能不是可行的')
        print('引理29.12 如果线性规划L没有可行解,则INITIALIZE-SIMPLEX返回“不可行”。否则,它返回一个基本解可行的合法松弛型')
        print('线性规划的基本定理')
        print('  特别地,任何线性规划都可能是不可行的,或是无界的,或有一个有限目标值的最优解；在每种情况下,SIMPLEX都能正确地起作用')
        print('定理29.13(线性规划的基本定理)以标准型给出任意的线性规划L可以是以下三者之一：')
        print(' 1) 有一个有限目标值的最优解')
        print(' 2) 不可行')
        print(' 3) 无界')
        print('如果L是不可行的,SIMPLEX返回“不可行”.如果L是无界的,SIMPLEX返回“无界”.否则,SIMPLEX返回一个有限目标值的最优解')
        print('练习29.5-1 伪代码实现INITIALIZE-SIMPLEX的第5行和第11行')
        print('练习29.5-2 证明：当INITIALIZE-SIMPLEX执行SIMPLEX的主循环时,永远不会返回“无界”')
        print('练习29.5-3 假设已知一个标准型的线性规划L,且假设对于L与L的对偶问题,其对应于初始松弛型的基本解都是可行的.',
            '证明:L的最优目标值是0')
        print('练习29.5-4 假设在线性规划中允许严格的不等式.证明:在这种情况下,线性规划的基本定理不再成立')
        print('练习29.5-5 用SIMPLEX解下列线性规划')
        print('练习29.5-6 略')
        print('练习29.5-7 P的1个变量的线性规划')
        print('思考题29-1 线性不等式的可行性')
        print('  已知一个在n个变量x1,x2,...,xn上的m个线性不等式的集合,',
            '线性不等式可行性问题询问是否有变量的一个设定能够同时满足每个不等式')
        print('  (a) 证明：如果有一个线性规划的算法,则可以利用它来解释线性不等式可行性问题',
            '在线性规划问题中用到的变量和约束的个数应该是n和m的多项式')
        print('  (b) 证明：如果有一个线性不等式可行性问题的算法,则可以用它来解决线性规划问题',
            '在线性不等式可行性问题中,用到的变量和线性不等式的个数应该是n和m的多项式,即线性规划中变量和约束的个数')
        print('思考题29-2 互补松弛性')
        print('  互补松弛描述原变量的值与对偶约束、对偶变量的值与原约束之间的关系')
        print('  证明：对任意的原线性规划和它相应的对偶问题,保持互补松弛性')
        print('思考题29-3 整数线性规划')
        print('  一个整数线性规划问题是一个加上变量x必须在整数上取值的额外约束的线性规划问题',
            '练习34.5-3说明仅确定一个整数线性规划是否可行解是NP-难度的,这表示问题不大可能有一个多项式时间的算法')
        print('  证明弱对偶性对整数线性规划成立')
        print('  证明对偶性对整数规划不总是成立')
        print('  已知一个标准型的原线性规划,定义P为原线性规划的最优目标值,D为其对偶问题的最优目标值',
            'IP为整数版本的原问题(原问题加上变量取整数值的约束)的最优目标值,ID为整数版本的对偶问题的最优目标值',
            '证明IP<=P=D<=ID')
        print('思考题29-4 Farkas引理')
        print('  令A为一个m*m矩阵,b为一个m维向量.则Farkas引理说明正好有一个系统Ax<=0,bx>0和yA=b,y>=0是可解的',
            '其中x是一个n维向量,y是一个m维向量.证明Farkas引理')
        # python src/chapter29/chapter29note.py
        # python3 src/chapter29/chapter29note.py

chapter29_1 = Chapter29_1()
chapter29_2 = Chapter29_2()
chapter29_3 = Chapter29_3()
chapter29_4 = Chapter29_4()
chapter29_5 = Chapter29_5()

def printchapter29note():
    """
    print chapter29 note.
    """
    print('Run main : single chapter twenty-nine!')
    chapter29_1.note()
    chapter29_2.note()
    chapter29_3.note()
    chapter29_4.note()
    chapter29_5.note()

# python src/chapter29/chapter29note.py
# python3 src/chapter29/chapter29note.py

if __name__ == '__main__':  
    printchapter29note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter30/chapter30note.py
# python3 src/chapter30/chapter30note.py
"""

Class Chapter30_1

Class Chapter30_2

Class Chapter30_3


"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np

class Chapter30_1:
    """
    chapter30.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter30.1 note

        Example
        ====
        ```python
        Chapter30_1().note()
        ```
        """
        print('chapter30.1 note as follow')
        print('第30章 多项式与快速傅里叶变换')
        print('两个n次多项式相加的花不达标方法所需的时间为Θ(n),而相乘的简单方法所需的时间为Θ(n^2)')
        print('在本章中,将快速傅里叶变换FFT方法是如何使多项式相乘的运行时间降低为Θ(nlgn)')
        print('傅里叶变换的最常见用途是信号处理，也是FFT最常见的用途，在时间域内给定的信号把时间映射到振幅的一个函数',
            '傅里叶分析允许将信号表示成各种频率的相移正弦曲线的一个加权总和')
        print('和频率相关联的权重和相位在频率域中刻画出信号的特性')
        print('在一个代数域F上，关于变量x的多项式定义为形式和形式表示的函数A(x)=∑ajxj')
        print('称值a0,a1,...,an-1为多项式的系数，所有系数都属于域F，典型的情况是复数集合C.如果一个多项式A(x)的最高次的非零系数为ak',
            '则称A(x)的次数(degree)是k.任何严格大于一个多项式次数的整数都是这个多项式的次数界.因此,对于次数界为n的多项式来说,其次数可以是0到n-1之间的任何整数',
            '也包括0和n-1在内')
        print('在多项式上可以定义各种运算,在多项式加法中,如果A(x)和B(x)是次数界为n的多项式,那么它们的和也是一个次数界为n的多项式C(x),',
            '并满足对所有属于定义域的x,都有C(x)=A(x)+B(x)')
        print('在多项式乘法中,如果A(x)和B(x)都是次数界为n的多项式,则说它们的乘积是一个次数界为2n-1的多项式积C(x),并满足对所有属于定义域的x,都有C(x)=A(x)B(x)')
        print('注意degree(C)=degree(A)+degree(B)蕴含degree-bound(C)=degree-bound(A)+degree-bound(B)-1<=degree-bound(A)+degree-bound(B)')
        print('但是不说C的次数界为A的次数界与B的次数界的和,这是因为如果一个多项式的次数界为k,也可以说该多项式的次数界为k+1')
        print('30.1 多项式的表示')
        # !从某种意义上说,多项式系数表示法与点值表示法是等价的
        print('从某种意义上说,多项式系数表示法与点值表示法是等价的,即用点值形式表示的多项式都对应唯一一个系数形式的多项式',
            '这两种表示结合起来，从而使这两个次数界为n的多项式乘法运算在Θ(nlgn)时间内完成')
        print('系数表示法')
        print('对一个次数界为n的多项式A(x)=∑ajxj来说,其系数表示法就是由一个由系数组成的向量a=(a0,a1,...,an-1)',
            '在本章所涉及的矩阵方程中,一般将它作为列向量看待')
        print('采用系数表示法对于某些多项式的运算是很方便的.例如对多项式A(x)在给定点x0的求值运算就是计算A(x0)的值',
            '如果使用霍纳法则,则求值运算的运行时间为Θ(n):')
        print('  A(x0)=a0+x0(a1+x0(a2+...+x0(an-2+x0(an-1))...))')
        print('类似地,对两个分别用系数向量a=(a0,a1,...,an-1)和b=(b0,b1,...,bn-1)表示的多项式进行相加时,所需的时间是Θ(n):',
            '仅输出系数向量c=(c0,c1,...,cn-1),其中对j=0,1,...,n-1,有cj=aj+bj')
        print('现在来考虑两个用系数形式表示的、次数界为n的多项式A(x)和B(x)的乘法运算,完成多项式乘法所需要的时间就是Θ(n^2)',
            '因为向量a中的每个系数必须与向量b中的每个系数相乘。当用系数形式表示时,多项式乘法运算似乎要比求多项式的值和多项式加法困难的多')
        print('卷积运算c=a＊b,多项式乘法与卷积的计算都是最基本的问题')
        print('点值表示法')
        print('  一个次数界为n的多项式A(x)的点值表示就是n个点值对所形成的集合：{(x0,y0),(x1,y1),...,(xn-1,yn-1)}')
        print('  其中所有xk各不相同,并且当k=0,1,...,n-1时有yk=A(xk)')
        print('  一个多项式可以有很多不同的点值表示,这是由于任意n个相异点x0,x1,...,xn-1组成的集合,都可以作为这种表示法的基础')
        print('  对于一个用系数形式表示的多项式来说,在原则上计算其点值表示是简单易行的,因为我们所要做的就是选取n个相异点x0,x1,...,xn-1',
            '然后对k=0,1,...,n-1,求出A(xk).根据霍纳法则,求出这n个点的值所需要的时间为Θ(n^2),在稍后可以看到,如果巧妙地选取xk的话,就可以加速这一计算过程,使其运行时间变为Θ(nlgn)')
        print('  求值计算的逆(从一个多项式的点值表示确定其系数表示中的系数)称为插值(interpolation).下列定理说明插值具有良定义,',
            '假设插值多项式的次数界等于已知的点值对的数目')
        print('定理30.1 (多项式插值的唯一性) 对于任意n个点值对组成的集合：{(x0,y0),(x1,y1),...,(xn-1,yn-1)},存在唯一的次数界为n的多项式A(x),',
            '满足yk=A(xk),k=0,1,...,n-1')
        print('要对n个点进行插值,还可以用另一种更快的算法,基于拉格朗日插值公式,拉格朗日插值公式等式右端是一个次数界为n的多项式',
            '可以在Θ(n^2)的运行时间内,运用拉格朗日公式来计算A的所有系数')
        print('n个点的求值运算与插值运算是良定义的互逆运算,它们将多项式的系数表示与点值表示进行相互转换,关于这些问题,上述算法的运行时间为Θ(n^2)')
        print('因此,对两个点值形式表示的次数界为n的多项式相加,所需时间为Θ(n)')
        print('如果已知两个扩充点值形式的输入多项式,使其相乘而得到点值形式的结果需要Θ(n)的时间,这要比采用系数形式的两个多项式相乘所需的时间少得多')
        print('考虑对一个点值表示的多项式,如何求其在某个新点上的值这一问题.对这个问题来说,最简单不过的方法显然就是先把该多项式转化为其系数形式,然后再求其在新点处的值')
        print('系数形式表示的多项式的快速乘法')
        print('  能否可以利用关于点值形式表示的多项式的线性时间乘法算法,来加快系数形式表示的多项式乘法运算的速度呢？',
            '答案依赖于能否快速把一个多项式从系数形式转换为点值形式(求值),和从点值形式转换为系数形式(插值)')
        print('可以采用需要的任何点作为求值点,但精心地挑选求值点,可以把两种表示法之间转化所需的时间压缩为Θ(nlgn)')
        print('如果选择“单位复根”作为求值点,则可以通过对系数向量进行离散傅里叶变换DFT,得到相应的点值表示')
        print('同样,也可以通过对点值对执行“逆DFT运算”,而获得相应的系数向量,这样就完成了求值运算的逆运算----插值',
            '可以在Θ(nlgn)的时间内执行DFT和逆DFT运算')
        print('两个次数界为n的多项式的积是一个次数界为2n的多项式。因此,在对输入多项式A和B进行求值之前,首先通过增加n个值为0的高阶系数,使其次数界增加到2n',
            '因为向量包含2n个元素,所以用到了2n次单位复根')
        print('如果已知FFT,就有下列运行时间为Θ(nlgn)的过程,该过程把两个次数界为n的多项式A(x)和B(x)进行乘法运算',
            '其中输入与输出均采用系数表示法。假定n为2的幂，通过加入为0的高阶系数,这个要求总能被满足：')
        print('(1) 使次数界增加一倍：通过加入n个值为0的高阶系数,把多项式A(x)和B(x)扩充为次数界为2n的多项式并构造其系数表示')
        print('(2) 求值：两次应用2n阶的FFT计算出A(x)和B(x)的长度为2n的点值表示。这两个点值表示中包含了两个多项式在2n次单位根处的值')
        print('(3) 点乘：把A(x)的值与B(x)的值逐点相乘,就可以计算出多项式C(x)=A(x)B(x)的点值表示,这个表示中包含了C(x)在每个2n次单位根处的值')
        print('(4) 插值：只要对2n个点值对应用一次FFT以计算出其逆DFT,就可以构造出多项式C(x)的系数表示')
        print('执行第1步和第3步所需时间为Θ(n),执行第2步和第4步所需时间为Θ(nlgn)')
        print('定理30.2 当输入与输出都采用系数形式来表示多项式时,就能够在Θ(nlgn)的时间内,计算出两个次数界为n的多项式的积')
        print('练习30.1-1 运用多项式乘法计算A(x)=7x^3-x^2+x-10和B(x)=8x^3-6x+3的乘积')
        print('练习30.1-2 求一个次数界为n的多项式A(x)在某已知点x0的值也可以用以下方法获得：把多项式A(x)除以多项式(x-x0)',
            '得到一个次数界为n-1的商多项式q(x)和余项r,并满足A(x)=q(x)(x-x0)+r',
            '显然A(x0)=r,试说明如何根据x0和A的系数,在Θ(n)的时间内计算出余项r以及q(x)中的系数')
        print('练习30.1-3 根据A(x)=∑ajx^j的点值表示推导出Arev(x)=∑an-1-jx^j的点值表示,假定没有一个点是0')
        print('练习30.1-4 证明：为了唯一确定一个次数界为n的多项式,n个相互不同的点值对是必需的,也就是说,如果给定少于n对不同的点值',
            '它们就无法确定唯一一个次数界为n的多项式')
        print('练习30.1-5 可以使用拉格朗日等式在Θ(n^2)的时间内进行插值运算')
        print('练习30.1-6 试解释在采用点值表示法时,用“显然”的方法来进行多项式除法错误在何处,即除以相应的y值',
            '试对除法有确切结果与无确切结果两种情况分别进行讨论')
        print('练习30.1-7 考察两个集合A和B,每个集合包含取值范围在0到10n之间的n个整数,要计算出A与B的笛卡尔和,它的定义如下：C={x+y,x∈A且y∈B}',
            '注意,C中整数的取值范围在0到20n之间.希望计算出C中的元素,并且求出C的每个元素可为A与B中元素和的次数。证明：解决这个问题需要Θ(nlgn)的时间')
        # python src/chapter30/chapter30note.py
        # python3 src/chapter30/chapter30note.py

class Chapter30_2:
    """
    chapter30.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter30.2 note

        Example
        ====
        ```python
        Chapter30_2().note()
        ```
        """
        print('chapter30.2 note as follow')
        print('DFT与FFT')
        print('如果使用单位复根的话,就可以在Θ(nlgn)时间内完成求值与插值运算')
        print('单位复根')
        print('  n次单位复根是满足w^n=1的复数w.n次单位复根刚好有n个,它们是e^(2∏ik/n),k=0,1,...,n-1')
        print('  欧拉公式e^(iu)=cos(u)+isin(u)')
        print('单位复根的性质')
        print('  (1) 相消引理')
        print('  (2) 折半引理')
        print('  (3) 求和引理')
        print('折半引理是递归的FFT算法的基础，1个元素的DFT就是该元素自身')
        print('递归的FFT算法RECURSIVE-FFT的运行时间,除了递归调用外,每条命令执行所需的时间为Θ(n)',
            'n为输入向量的长度.因此,关于运行时间有下列递归式:')
        print('  T(n)=2T(n/2)+Θ(n)=Θ(nlgn)')
        print('因此,运用快速傅里叶变换,可以在Θ(nlgn)的时间内,求出次数界为n的多项式在n次单位复根处的值')
        print('对单位复根进行插值')
        print('  把一个多项式从点值表示转化成系数表示,进而完成多项式乘法方案.按如下方式进行插值：',
            '把DFT写成一个矩阵方程,然后再检查其逆矩阵的形式')
        print('  可以把DFT写成矩阵积y=Vna,其中Vn是由wn的适当幂组成的一个范德蒙德矩阵')
        print('定理30.7 对j,k=0,1,...,n-1,Vn^-1的(j,k)处的元素为wn^(-kj)/n')
        print('定理30.8 (卷积定理)对任意两个长度为n的向量a和向量b,其中n是2的幂',
            'a＊b=DFT^(-1)(2n)(DFT2n(a)·DFT2n(b))',
            '其中向量a和b用0扩充使其长度达到2n,“·”表示2个2n个元素组成的向量的点乘')
        print('练习30.2-1 略')
        print('练习30.2-2 计算向量(0,1,2,3)的DFT')
        print('练习30.2-3 使用运行时间在Θ(nlgn)的方案重做练习30.1-1')
        print('练习30.2-4 写出在Θ(nlgn)的运行时间内计算出DFT^(-1)n的伪代码')
        print('练习30.2-5 试着把FFT过程推广到n是3的幂的情形,写出其运行时间的递归式并求解该式')
        print('练习30.2-6 假定不是在复数域上执行n个元素的FFT(n为偶数),而是在整数模m所生成的环Zm上执行FFT',
            '其中m=2^(tn/2)+1,并且t是任意正整数.对模m,用w=2^t来代替wn作为主n次单位根.证明:在该系统中DFT与逆DFT有良定义')
        print('练习30.2-7 已知一组值z0,z1,...,zn-1(可能有重复),说明如何求出仅在z0,z1,...,zn-1处(可能有重复)值为0的次数界为n的多项式P(x)的系数',
            '所给出的过程的运行时间应该是O(nlg^2n),(提示：当且仅当P(x)是(x-zj)的倍数时,多项式P(x)在zj处的值为0)')
        print('练习30.2-8 DFT是线性调频变换的一种特殊情况(z=wn).证明:对任意复数z,可以在O(nlgn)的时间内求出线性调频变换的值',
            '可以把线性调频变换看作为卷积')
        # python src/chapter30/chapter30note.py
        # python3 src/chapter30/chapter30note.py

class Chapter30_3:
    """
    chapter30.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter30.3 note

        Example
        ====
        ```python
        Chapter30_3().note()
        ```
        """
        print('chapter30.3 note as follow')
        print('30.3 有效的FFT实现')
        print('由于DFT的实际应用(如信号处理)需要极高的速度,所以本节将讨论两种有效的FFT实现方法.')
        print('运行时间为Θ(nlgn)的FFT算法的迭代实现方法隐含的常数要比递归实现方法中的常数小',
            '将深入分析迭代实现方法,设计出一个有效的并行FFT电路')
        print('ITERATIVE-FFT中要用到BIT-REVERSE-COPY算法,位转置算法')
        print('调用BIT-REVERSE-COPY的运行时间当然是Θ(nlgn),因为迭代了n次,并且可以在O(lgn)时间内,',
            '对一个在0到n-1之间的lgn位整数进行反向操作(在实际应用中,通常事先就知道了n的初值,所以可以计算出一张表,求出每个k的rev(k))',
            '使BIT-REVERSE-COPY的运行时间为Θ(n),且该式中隐含的常数也较小')
        print('并行FFT电路')
        print('  可以利用使得能够有效实现迭代FFT算法的许多性质,来产生一个有效的并行FFT算法,',
            '可以将并行FFT算法表示成一个与比较网络相似的电路。FFT电路使用蝴蝶操作而不是比较器')
        print('  关于n个输入的FFT的PARALLEL-FFT电路.电路一开始就对输入进行位反转置换,其后的电路分为lgn级,每一级由n/2并行执行的蝴蝶操作所组成',
            '因此电路的深度为Θ(nlgn)')
        print('  电路PARALLEL-FFT的最左边的部分执行位反转置换，其余部分模拟迭代的ITERATIVE-FFT过程',
            '因为最外层for循环的每次迭代均执行n/2次独立的蝴蝶操作,所以电路并行地执行它们,在过程ITERATIVE-FFT中每次迭代的值s对应于图中的一级蝴蝶',
            '在第s级中(s=1,2,...,lgn),有n/2^s组蝴蝶(对应于ITERATIVE-FFT中k的每个值),每组中有2^(s-1)个蝴蝶(对应于ITERATIVE-FFT中j的每个值)',
            '图中所示的蝴蝶对应于最内层循环,蝴蝶中用到的旋转因子对应于ITERATIVE-FFT中用到的那些旋转因子')
        print('练习30.3-1 试说明如何用过程ITERATIVE-FFT计算出输入向量(0,2,3,-1,4,5,7,9)的DFT')
        print('练习30.3-2 试说明如何把位反转置换放在计算过程的最后而不是在开始处,以实现FFT算法.(提示：考虑逆DFT)')
        print('练习30.3-3 ITERATIVE-FFT在每级中计算旋转因子多少次，重写ITERATIVE-FFT,使其在阶段s中只计算旋转因子2^(s-1)次')
        print('练习30.3-4 假设FFT电路中的加法器有时会发生错误:不论输入如何,它们的输出总是为0.假定确有一个加法器发生上述情况',
            '描述如何能够通过给整个FFT电路提供输入值并观察其输出,来找到那个产生错误的加法器')
        print('思考题30-1 分治算法')
        print('a) 说明如何仅用三次乘法,就能求出线性多项式ax+b与cx+d的乘积,有一个乘法运算是(a+b)·(c+d)')
        print('b) 试写出两种分治算法,使其在Θ(n^lg3)的运行时间内,求出两个次数界为n的多项式的乘积。第一个算法把输入多项式的系数分成高阶系数与低阶系数各一半',
            '第二个算法根据其系数下标的奇偶性来进行划分')
        print('c) 证明：用O(n^lg3)步可以计算出两个n比特的整数的乘积,其中每一步至多对固定数量1比特的值进行操作')
        print('思考题30-2 Toeplitz矩阵')
        print('  Toeplitz矩阵是一个满足如下条件的n*n矩阵A=(aij):aij=ai-1,j-1; i=2,3,...,n; j=2,3,...,n')
        print('a) 两个Toelitz矩阵的和是一定是Toeplitz矩阵，积也是Toelitz矩阵')
        print('b) 试说明如何表示Toeplitz矩阵,才能在O(n)时间内求出两个n*nToeplitz矩阵的和')
        print('c) 写出一个运行时间为O(nlgn)的算法,使其能够计算出n*nToeplitz矩阵与n维向量的乘积')
        print('d) 写出一个高效算法,使其能够计算出两个n*nToeplitz矩阵的乘积,并分析算法的运行时间')
        print('思考题30-3 多维快速傅里叶变换')
        print('  可以将1维的离散傅里叶变换推广到d维上')
        print('a) 证明可以通过依次在每个维上计算1维的DFT,来计算一个d维的DFT')
        print('b) 证明维的次序并无影响,因此可以通过在任意次序的d维中计算1维DFT来计算1个d维的DFT')
        print('c) 证明如果通过计算快速傅里叶变换来计算每1维的DFT,则计算一个d维的DFT的总时间是O(nlgn),与d无关')
        print('思考题30-4 求多项式在某一点的所有阶导数的值')
        print('  证明：可以在O(nlgn)的时间内,求出A(x)的所有非平凡单数在x0处的值')
        print('思考题30-5 多项式在多个点的求值')
        print('  运用霍纳(Horner)法则,就能够在O(n)的时间内,求出次数界为n-1的多项式在单个点的值',
            '运用FFT也能够在O(nlgn)的时间内,求出多项式在所有n个单位复根处的值.',
            '可以在O(nlg^2(n))的时间内,求出一个次数界为n的多项式在任意n个点的值')
        print('思考题30-6 运用模运算的FFT')
        print('  离散傅里叶变换(DFT)要求使用复数,因此,由于舍入误差而导致精确性下降.对某些问题来说,已知其答案仅包含整数,并且为了保证准确地计算出答案',
            '要求我们利用基于模运算的一种FFT的变异。例如求两个整系数的多项式对的积的问题就属于这类问题',
            '即运用一个长度为Ω(n)位的模来处理n个点的DFT.')
        print('a) 假定我们寻找最小的k,使p=kn+1为质数。证明：我们预计k约为lgn(k的值可能比lgn大一些或小一些),但我们能够',
            '合理地预计出k的O(lgn)个候选值的平均值。p的预计长度与n的长度有什么关系')
        # python src/chapter30/chapter30note.py
        # python3 src/chapter30/chapter30note.py

chapter30_1 = Chapter30_1()
chapter30_2 = Chapter30_2()
chapter30_3 = Chapter30_3()

def printchapter30note():
    """
    print chapter30 note.
    """
    print('Run main : single chapter thirty!')
    chapter30_1.note()
    chapter30_2.note()
    chapter30_3.note()

# python src/chapter30/chapter30note.py
# python3 src/chapter30/chapter30note.py

if __name__ == '__main__':  
    printchapter30note()
else:
    pass

```

```py

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def fft_demo():
    x = np.arange(-100, 100, 0.5)
    y = np.sin(x) + np.sin(3 * x)
    plt.figure()
    plt.plot(x, y)
    plt.show()
    plt.figure()
    plt.plot(fft.fftfreq(x.shape[-1]), abs(fft.fft(y)))
    plt.show()
    plt.imshow(np.sin(np.outer(x, x)))
    plt.show()

if __name__ == '__main__':  
    fft_demo()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter31/chapter31note.py
# python3 src/chapter31/chapter31note.py
"""

Class Chapter31_1

Class Chapter31_2

Class Chapter31_3

Class Chapter31_4

Class Chapter31_5

Class Chapter31_6

Class Chapter31_7

Class Chapter31_8

Class Chapter31_9

"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np

if __name__ == '__main__':
    import numtheory as _nt
else:
    from . import numtheory as _nt

class Chapter31_1:
    """
    chapter31.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.1 note

        Example
        ====
        ```python
        Chapter31_1().note()
        ```
        """
        print('chapter31.1 note as follow')
        print('第31章 有关数论的算法')
        print('数论一度被认为是漂亮但是却没什么大用处的纯数学学科。有关数论的算法被广泛使用，部分是因为基于大素数的密码系统的范明')
        print('系统的安全性在于大素数的积难于分解，本章介绍一些初等数论知识和相关算法')
        print('31.1节介绍数论的基本概念，例如整除性、同模和唯一因子分解等')
        print('31.2节研究一个世界上很古老的算法；关于计算两个整数的最大公约数的欧几里得算法')
        print('31.3节回顾模运算的概念')
        print('31.4节讨论一个已知数a的倍数模n所得到的集合,并说明如何利用欧几里得算法求出方程ax=b(modn)的所有解')
        print('31.5节阐述中国余数定理')
        print('31.6节考察已知数a的幂模n所得的结果，并阐述一种已知a,b,n,可以有效计算a^b模n')
        print('31.7节描述RSA公开密钥加密系统')
        print('31.8节主要讨论随机性素数基本测试')
        print('31.9回顾一种把小整数分解因子的简单而有效的启发性方法,分解因子是人们可能想到的一个难于处理的问题',
            '这也许是因为RSA系统的安全性取决于对大整数进行因子分解的困难程度')
        print('输入的规模与算数运算的代价')
        print('  因为需要处理一些大整数,所以需要调整一下如何看待输入规模和基本算术运算的代价的看法',
            '一个“大的输入”意味着输入包含“大的整数”,而不是输入中包含“许多整数”(如排序的情况).',
            '因此,将根据表示输入数所要求的的位数来衡量输入的规模,而不是仅根据输入中包含的整数的个数',
            '具有整数输入a1,a2,...,ak的算法是多项式时间算法,仅当其运行时间表示lga1,lga2,...,lgak的多项式,',
            '即它是转换为二进制的输入长度的多项式')
        print('  发现把基本算术运算(乘法、除法或余数的计算)看作仅需一个单位时间的原语操作是很方便的',
            '但是衡量一个数论算法所需要的位操作的次数将是比较适宜的，在这种模型中，用普通的方法进行两个b位整数的乘法',
            '需要进行Θ(b^2)次位操作.')
        print('一个b位整数除以一个短整数的运算,或者求一个b位整数除以一个短整数所得的余数的运算,也可以用简单算法在Θ(b^2)的时间内完成',
            '目前也有更快的算法.例如,关于两个b位整数相乘这一运算,一种简单分治算法的运行时间为Θ(b^lg2(3))',
            '目前已知的最快算法的运行时间为Θ(blgblglgb),在实际应用中,Θ(b^2)的算法常常是最好的算法,将用这个界作为分析的基础')
        print('在本章中,在分析算法时一般既考虑算术运算的次数,也考虑它们所要求的位操作的次数')
        print('31.1 初等数论概念')
        print('整数性和约数')
        print('  一个整数能被另一个整数整除的概念是数论中的一个中心概念。记号d|a(d整数a),意味着对某个整数k,有a=kd',
            '0可被任何整数整除.如果a>0且d|a,则|d|<=|a|.如果d|a,则也可以说a是d的倍数。')
        print('素数和合数')
        print('  对于某个整数a>1，如果它仅有平凡约数1和a,则称a为素数(或质数)。素数具有许多特殊性质,在数论中起着关键作用.按顺序看,前20个素数',
            '2,3,5,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71')
        print('  证明：有无穷多个素数。不是素数的整数a>1称为合数。例如，因为有3|39,所以39是合数。整数1被称为基数，他既不是素数也不是合数',
            '类似地,整数0和所有负整数既不是素数，也不是合数')
        print('除法定理，余数和同模')
        print('  已知一个整数n,所有整数都可以划分是n的倍数的整数,以及不是n的倍数的整数。对于不是n的倍数的那些整数',
            '又可以根据它们除以n所得的余数来进行分类,数论的大部分理论都是基于上述划分的')
        print('定理31.1(除法定理) 对任意整数a和任意正整数n,存在唯一的整数q和r,满足0<=r<n,并且a=qn+r',
            '值q=[a/n]称为除法的商.值r=a mod n称为除法的余数。n|a当且仅当a mod n=0')
        print('根据整数模n所得的余数,可以把整数分成n个等价类.包含整数a的模n等价类为：')
        print('  [a]n={a+kn : k∈Z}')
        print(' 如果用0表示[0]n,用1表示[1]n等等,每一类均用其最小的非负元素来表示,',
            '提到Zn的元素-1就是指[n-1]n,因为-1=n-1(mod n)')
        print('公约数与最大公约数')
        print('  如果d是a的约束并且也是b的约数,则d是a与b的公约数.例如30的约数为1,2,3,5,6,10,15,30,因此24与30的公约数为1,2,3和6',
            '注意，1是任意两个整数的公约数')
        print('  公约数的一条重要性质为：d|a并且d|b蕴含着d|(a+b)并且d|(a-b)')
        print('  更一般地，对任意整数x和y,有d|a并且d|b蕴含着d|(ax+by);同样,如果a|b,则或者|a|<=|b|,或者b=0,这说明:a|b且b|a蕴含着a=±b',
            '两个不同时为0的整数a与b的最大公约数表示成gcd(a,b),例如gcd(24,30)=6,gcd(5,7)=1,gcd(0,9)=9',
            '如果a与b不同时为0,则gcd(a,b)是一个在1与min(|a|,|b|)之间的整数,定义gcd(0,0)=0')
        print('下列性质是gcd函数的基本性质:')
        print('  gcd(a,b)=gcd(b,a)')
        print('  gcd(a,b)=gcd(-a,b)')
        print('  gcd(a,b)=gcd(|a|,|b|)')
        print('  gcd(a,0)=|a|')
        print('  gcd(a,ka)=|a|,对任何k∈Z')
        print('定理31.2 如果a和b是不都为0的任意整数,则gcd(a,b)是a与b的线性组合集合{ax+by:x,z∈Z}中的最小正元素')
        print('推论31.3 对任意整数a与b,如果d|a并且d|b,则d|gcd(a, b)')
        print('推论31.4 对所有整数a和b以及任意非负整数n,gcd(an,bn)=ngcd(a,b)')
        print('互质数')
        print('  如果两个整数a与b仅有公因数1,即如果gcd(a,b)=1,则a与b称为互质数,则它们的积与p互为质数')
        print('定理31.6 对任意整数a,b和p,如果gcd(a, p)=1且gcd(b,p)=1,则gcd(ab,p)=1')
        print('唯一的因子分解')
        print('定理31.7 对所有素数p和所有整数a,b,如果p|ab,则p|a或p|b(或两者都成立)')
        print('定理31.8 (唯一质因子分解) 合数a仅能以一种方式,写成如下的乘积形式')
        print('练习31.1-1 证明有无穷多个素数。(提示：证明素数p1,p2,...,pk都不能整除(p1p2...pk)+1)')
        print('练习31.1-2 证明：如果a|b且b|c,则a|c')
        print('练习31.1-3 证明：如果p是素数并且0<k<p,则gcd(k, p)=1')
        print('练习31.1-4 证明推论31.5')
        print('练习31.1-5 证明：如果p是素数且0<k<p.证明对所有整数a,b和素数p')
        print('练习31.1-6 证明：如果a和b是任意整数,且满足a|b和b>0,则对任意x有：(x mod b) mod a == x mod a')
        print('练习31.1-7 对任意整数k>0,如果存在一个整数a满足a^k=n,则说整数n为k次幂',
            '如果对于某个整数k>1,n>1是一个k次幂,则说n是非平凡幂.说明如何关于b的多项式时间内,确定出一个b位整数n是非平凡幂')
        print('练习31.1-8 略')
        print('练习31.1-9 证明：gcd运算满足结合律。亦即证明对所有整数a,b,c,有gcd(a,gcd(b,c))=gcd(gcd(a,b),c)')
        print('练习31.1-10 证明定理31.8 ')
        print('练习31.1-11 试写出一个b位整数除以一个短整数的有效算法,以及求一个n位整数除以一个短整数的余数的有效算法.所给出的算法的运行时间应为O(b^2)')
        print('练习31.1-12 写出一个能把一个b位二进制数转化为相应的十进制表示的有效算法.论证：如果长度至多为b的整数的乘法与除法运算所需时间为M(b),',
            '则执行二进制到十进制转换所需的时间为Θ(M(b)lgb).(运用分治法,把数分为顶部和底部两部分,分别进行递归操作而获得所需结果)')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_2:
    """
    chapter31.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.2 note

        Example
        ====
        ```python
        Chapter31_2().note()
        ```
        """
        print('chapter31.2 note as follow')
        print('31.2 最大公约数')
        print('运用欧几里得算法有效地计算出两个整数的最大公约数,在对其运行时间分析中,会发现与斐波那契数存在着联系,由此可获得欧几里得算法在最坏情况下的输入')
        print('在本节中仅限于对非负整数的情况讨论gcd(a, b) = gcd(|a|, |b|)')
        print('原则上讲,可以根据a和b的素数因子分解,求出正整数a和b的最大公约数gcd(a,b)')
        print('如果a=p1^e1 p2^e2 p3^e3...pr^er; a=p1^f1 p2^f2 p3^f3...pr^fr;')
        print('其中使用了零指数,使得素数集合p1,p2,...,pr对于a和b相同')
        print('  gcd(a,b)=p1^min(e1,f1) p2^min(e2,f2) ...pr^min(er,fr)')
        print('目前已知的最好的分解因子算法也不能达到多项式的运行时间,因此,根据这种方法来计算最大公约数,不大可能获得一种有效的算法')
        print('定理31.9 (GCD递归定理) 对任意负整数a和任意的正整数b')
        print(' gcd(a,b)=gcd(b, a mod b)')
        print('欧几里得算法')
        print(_nt.euclid(30, 21))
        print('欧几里得算法的运行时间')
        print('  最坏情况下,可以把欧几里得算法看成输入a与b的大小的函数。不失一般性,假定a>b>=0',
            '这个假设的合理性是基于下述观察的:如果b>a>=0,则EUCLID(a,b)立即会递归调用EUCLID(b, a),即如果第一个自变量小于第二个自变量',
            '则EUCLID进行一次递归调用以使两个自变量兑换,然后继续往下执行.类似地,如果b=a>0,则过程在进行一次递归调用后就终止执行,因为a mod b=0')
        print('引理31.10 如果a>b>=1并且EUCLID(a, b)执行了k>=1次递归调用,则a>=Fk+2,b>=Fk+1')
        print('定理31.11 (Lame定理) 对任意整数k>=1,如果a>b>=1且b<Fk+1,则EUCLID(a, b)的递归调用次数少于k次')
        print('由于Fk约为v^k/sqrt(5),其中v是定义的黄金分割率(1+sqrt(5))/2,所以EUCLID执行中的递归调用次数O(lgb)',
            '如果过程EUCLID作用于两个b位数,则它执行O(b)次算术和O(b^3)次位操作')
        print('欧几里得算法的推广形式')
        print('  现在来重写欧几里得算法以计算出其他的有用信息.特别地,推广该算法,使它能计算出满足下列条件的整系数x和y:d=gcd(a, b)=ax+by')
        print('  注意,x与y可能为0或负数.以后会发现这些系数对计算模乘法的逆是非常有用的,过程EXTENDED-EUCLID的输入为一对非负整数,返回一个满足上式的三元式(d,x,y)')
        print(_nt.euclid(30, 21))
        print('练习31.2-1 略')
        print('练习31.2-2 答案如下')
        print(_nt.extend_euclid(899, 493))
        print('练习31.2-3 证明对所有整数a, k, n; gcd(a, n) = gcd(a+kn, n)')
        print('练习31.2-4 仅用常量大小的存储空间(即仅存储常数个整数值)把过程EUCLID改写成迭代形式')
        print('练习31.2-5 如果a>b>=0,证明EUCLID(a, b)至多执行了1+logvb次递归调用.把这个界改进为1+logv(b / gcd(a, b))')
        print('练习31.2-6 过程EXTEND-EUCLID(Fk+1, Fk)返回1')
        print(_nt.extend_euclid(11, 8))
        print(_nt.extend_euclid(19, 11))
        print('练习31.2-7 用递归等式gcd(a0, a1,...,an)=gcd(a0, gcd(a1,...,an))来定义多余两个变量的gcd函数',
            '证明gcd函数的返回值与其自变量的次序无关,说明如何找出满足gcd(a0,a1,...,an)=a0x0+a1x1+..+anxn的整数x0,x1,...,xn',
            '证明所给出的算法执行的除法运算次数为O(n+lg(max{a0,a1,...,an}))')
        print('练习31.2-8 定义1cm(a0,a1,...,an)是n个整数a1,a2,...,an的最小公倍数,即每个ai的倍数中的最小非负整数',
            '说明如何用(两个自变量)gcd函数作为子程序以有效地计算出1cm(a1,a2,...,an)')
        print('练习31.2-9 证明n1,n2,n3和n4是两两互质的当且仅当gcd(n1n2,n3n4)=gcd(n1n3,n2n4)=1',
            '更一般地,证明n1,n2,...,nk是两两互质的,当且仅当从ni中导出的[lgk]对数互为质数')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_3:
    """
    chapter31.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.3 note

        Example
        ====
        ```python
        Chapter31_3().note()
        ```
        """
        print('chapter31.3 note as follow')
        print('31.3 模运算')
        print('可以把模运算非正式地与通常的整数运算一样看待,如果执行模n运算,则每个结果值x都由集合{0,1,...,n-1}中的某个元素所取代',
            '该元素在模n的意义下与x等价(即用x mod n来取代x).如果仅限于运用加法、减法和乘法运算,则用这样的非正式模型就足够了.',
            '模运算模型最适合于用群论结构来进行描述')
        print('有限群')
        print('  群(S,+)是一个集合S和定义在S上的二进制运算+,它满足下列性质')
        print('  1) 封闭性：对所有a,b∈S,有a+b∈S,')
        print('  2) 单位元：存在一个一个元素e∈S,称为群的单位元,满足对所有a∈S,e+a=a+e=a')
        print('  3) 结合律：对所有a,b,c∈S,有(a+b)+c=a+(b+c)')
        print('  4) 逆元：对每个a∈S,存在唯一的元素b∈S,称为a的逆元,满足a+b=b+a=e')
        print('根据模加法与模乘法所定义的群')
        print('子群')
        print('  如果(S,+)是一个群,S‘∈S,并且(S’,+)也是一个群,则(S’,+)称为(S,+)的子群',
            '例如,在加法运算下,偶数形成一个整数的子群.下列定理提供了识别子群的一个有用的工具')
        print('定理31.14(一个有限群的非空封闭子集是一个子群) 如果(S,+)是一个有限群,S`是S的一个任意非空子集并满足：',
            '对所有a,b∈S’。则(S\',+)是(S,+)的一个子群')
        print('定理31.15(拉格朗日定理) 如果(S,+)是一个有限群,(S’,+)是(S,+)的一个子群,则|S‘|是|S|的一个约数')
        print('推论31.16如果S’是有限群S的真子群,则|S‘|<=|S|/2')
        print('由一个元素生成的子群')
        print('  定理31.17 对任意有限群(S,+)和任意a∈S,一个元素的阶等于它所生成的子群的规模,即ord(a)=|<a>|')
        print('  推论31.18 序列a(1),a(2),...是周期性序列,其周期为t=ord(a);即a(i)=a(j)当且仅当i=j(mod t)')
        print('  推论31.19 如果(S,+)是一个具有单位元e的有限群,则对所有a∈S')
        print('练习31.3-1 画出群(Z4,+4)和群(Z5*,*5)的运算表。通过找这两个群的元素间的、满足')
        print('练习31.3-2 证明定理31.14')
        print('练习31.3-3 证明：如果p是素数且e是正整数,则phi(e^e)=p^(e-1)(p - 1)')
        print('练习31.3-4 证明：对任意n>1和任意a∈Zn,由式fa(x)=ax mod n所定义的函数fa:Zn->Zn是Zn的一个置换')
        print('练习31.3-5 列举出Z9和Z13的所有子群')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_4:
    """
    chapter31.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.4 note

        Example
        ====
        ```python
        Chapter31_4().note()
        ```
        """
        print('chapter31.4 note as follow')
        print('31.4 求解模线性方程')
        print('考虑求解下列方程的问题:ax=b(mod n)')
        print('其中a>0,n>0.这个问题有若干种应用;例如RSA公钥密码系统中,寻找密钥过程的一部分。假设已知a,b和n,希望求出所有满足式的对模n的x值',
            '可能没有解,也可能有一个或多个解')
        print('设<a>表示由a生成的Zn的子群.由于<a>={a(x):x>0}={ax mod n:x>0},所以方程有一个解当且仅当b∈<a>。',
            '拉格朗日定理告诉我们,|<a>|必定是n的约数,下列定义准确地刻画了<a>的特性')
        print('定理31.20 对任意整数a和n,如果d=gcd(a,n),则在Zn中')
        print('推论31.21 方程ax=b (mod n)对于未知量x有解,当且仅当gcd(a,n)|b')
        print('推论31.22 方程ax=b (mod n)或者对模n有d个不同的解,其中d=gcd(a, n),或者无解')
        print('定理31.23 设d=gcd(a,n),假定对整数x‘和y’,有d+ax‘+ny‘。如果d|b,则方程ax=b(mod b)有一个解的值为x0,满足',
            'x0=x’(b/d) mod n')
        print('定理31.24 假设方程ax=b (mod n)有解(即有d|b,其中d=gcd(a,n))x0是该方程的任意一个解,',
            '则该方程对模n恰有d个不同的解,分别为:xi=x0+i(n/d0)(i=1,2,...,d-1)')
        print('MODULAR-LINEAR-EQUATION-SOLVER执行O(lgn+gcd(a, n))次算术运算.因为EXTENDED-EUCLID需要执行O(lgn)次算术运算')
        print('推论31.25 对任意n>1,如果gcd(a, n)=1,则方程ax=b(mod n)对模n有唯一解')
        print('推论31.26 对任意n>1,如果gcd(a, n)=1,则方程ax=1(mod n)对模n有唯一解')
        print('练习31.4-1 求出方程35x=10(mod 50)的所有解')
        _nt.modular_linear_equation_solver(35, 10, 50)
        print('练习31.4-2 证明：当gcd(a, n)=1,由方程ax=ay(mod n)可得x=y(mod n).通过一个反例gcd(a, n)>1的情况来证明条件gcd(a,n)=1是必要的')
        print('练习31.4-3 考察下列对过程MODULAR-LINEAR-EQUATION-SOLVER的第3行修改')
        print('练习31.4-4 略')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_5:
    """
    chapter31.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.5 note

        Example
        ====
        ```python
        Chapter31_5().note()
        ```
        """
        print('chapter31.5 note as follow')
        print('31.5 中国余数定理')
        print('大约在公元100年,中国的数学家孙子解决了一下问题:找出被3,5和7除时余数分别为2,3和2的所有整数x.有一个解为x=23;',
            '所有的解是形如23+105k(k为任意整数)的整数。\"中国余数定理\"提出,对一组两两互质的整数')
        print('中国余数定理:对一组两两互质的模数(如3,5和7)来说,其取模运算的方程组与对其积(如105)取模运算的方程之间存在着一种对应关系')
        print('中国余数定理有两个主要作用。设整数n因式分解为n=n1n2...nk,其中因子ni两两互质',
            '首先,中国余数定理是一个描述性的“结构定理”,说明Zn的结构等同于笛卡尔积Zn1×Zn2×...×Znk的结构')
        print('其中,第i个组元定义了对模ni的组元之间的加法与乘法运算',
            '其次,用这种描述常常可以获得有效的算法,因为处理Zn系统中的每个系统可能比处理模n运算效率更高(从位操作次数来看)')
        print('定理31.27(中国余数定理)设n=n1n2...nk,其中因子ni两两互质.考虑下列对应关系：a<->(a1,a2,...,ak)')
        print(' 其中a∈Zn,ai∈Zni,而且对i=1,2,...,k. ai = a mod ni')
        print('对Zn中元素所执行的运算可以等价地作用于对应的k元组,即在适当的系统中独立地对每个坐标位置执行所需的运算')
        print(' a<->(a1,a2,...,ak)')
        print(' b<->(b1,b2,...,bk)')
        print('推论31.28 如果n1,n2,...,nk两两互质,n=n1n2...nk,则对任意整数a1,a2,...,ak(i=1,2,...,k),方程组x=ai(mod ni)')
        print('推论31.29 如果n1,n2,...,nk两两互质,n=n1n2...nk,则对所有整数x和a(i=1,2,...,k)')
        print('练习31.5-1 计算出使方程x=4(mod 5)和x=5(mod 11)同时成立的所有解')
        print('练习31.5-2 试找出被9,8,7除时,余数分别为1,2,3的所有整数x')
        print('练习31.5-3 论证:在定理31.27的定义下,如果gcd(a, n)=1')
        print('练习31.5-4 证明:对于任意的多项式f,方程f(x)=0 (mod n)的根的数目等于每个方程',
            'f(x)=0 (mod n1),f(x)=0 (mod n2),...,f(x)=0(mod nk)的根的数目的积')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_6:
    """
    chapter31.6 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.6 note

        Example
        ====
        ```python
        Chapter31_6().note()
        ```
        """
        print('chapter31.6 note as follow')
        print('31.6 元素的幂')
        print('正如考虑一个已知元素a对模n的倍数一样,常常自然地考虑对模n的a的幂组成的序列,其中a∈Zn:a0,a1,a2,a3,...')
        print('定理31.30 (欧拉定理) 对于任意整数n>1,a^phi(n)=1 (mod n)对所有a∈Zn都成立')
        print('定理31.31 (费马定理) 如果p是素数,则a^(p-1)=1 (mod p)对所有a∈Zp都成立')
        print('定理31.32 对所有素数p>2和所有正整数e,满足Zn为循环群的n(n > 1)值为2,4,p^e,2p^e')
        print('定理31.33 (离散对数定理) 如果g是Zn*的一个原根,则等式g^x=g^y(mod n)成立,当且仅当等式x=y(mod phi(n))成立')
        print('定理31.34 如果p是一个奇素数且e>=1,则方程x^2=1(mod p^e)')
        print('推论31.35 如果对模n存在1的非平凡平方根,则n是合数')
        print('运用反复平方法求数的幂')
        print('  数论计算中经常出现一种运算,就是求一个数的幂对另外一个数的模的运算,也称为模取幂',
            '更准确地说,希望找出一种有效的方法来计算a^b mod n的值,其中a,b为非负整数,n为正整数',
            '在许多素数测试子程序和RSA公开密钥加密系统中,模取幂运算是一种很重要的运算',
            '当用二进制来表示b时,采用反复平方法,可以有效地解决这个问题')
        print('  设<bk,bk-1,...,b1,b0>是b的二进制表示.(亦即,二进制表示有k+1位长,bk为最高有效位,b0为最低有效位))')
        print('  下列过程随着c的值从0到b成倍增长,最终计算出a^c mod n')
        print('练习31.6-1 试画出一张表以说明Z11中每个元素的阶.找出最小的原根g并计算出一张表,要求写出所有x∈Z11,相应的ind11.g(x)的值')
        print('练习31.6-2 写出一个模取值幂算法,要求该算法检查b的各位的顺序为从右向左,而不是从左向右')
        print('练习31.6-3 假设已知phi(n),试说明如何运用过程MODULAR-EXPONENTIATION计算出对任意a∈Zn,a^(-1) mod n')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_7:
    """
    chapter31.7 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.7 note

        Example
        ====
        ```python
        Chapter31_7().note()
        ```
        """
        print('chapter31.7 note as follow')
        print('RSA公钥加密系统')
        print('公钥加密系统可以对传输于两个通信单位之间的消息进行加密,这样即使窃听者听到被加密的消息,也不能对加密消息进行破译',
            '公钥加密系统还能够使通信的一方,在电子报文的末尾附加一个无法伪造的“数字签名”')
        print('RSA公钥加密系统主要基于以下事实:寻求大素数是很容易的,但要把一个数分解为两个大素数的积确实相当困难的')
        print('公钥加密系统')
        print('  在公钥加密系统中,每个参与者都用一把公钥和一把密钥.没把密钥都是一条信息，例如在RSA公密钥加密系统中,每个密钥均是由一对整数组成')
        print('  在密码学中常把Alice和Bob作为例子：用P_A和S_A分别表示Alice的公钥和密钥;用P_B和S_B分别表示Bob的公钥和密钥')
        print('  每个参与者均自己创建起公钥和密钥。密钥需要保密，但公钥则可以对任何人公开或干脆公之于众。事实上，如果每个参与者的公钥都能在一个公开目录中查到的话是很方便的',
            '这样能使任何参与者容易地获得任何其他参与者的公钥')
        print('  公钥和密钥指定可适用于任何信息的功能函数.设D表示允许的信息集合。例如,D可能是所有有限长度的位序列集合')
        print('  在最简单的、原始的公钥密码学中，要求公钥与密钥说明一种从D到其自身的一一对应函数.对应于Alice的公钥P_A的函数用P_A()表示',
            '对应于她的密钥S_A的函数表示成S_A(),因此P_A()与S_A()函数都是D的排列.假定如果已知密钥P_A或S_A,就能够有效地计算出函数P_A()和S_A()')
        print('  任何参与者的公钥和密钥都是一个匹配对,他们制定的函数互为反函数,亦即对任何消息M∈D,有M=S_A(P_A(M));M=P_A(S_A(M))')
        print('在RSA公钥加密系统中,重要的是除Alice外,没有人能在较短时间内计算出S_A().送给Alice加密邮件的保密程度与Alice数组签名的真实性均依赖于以下假设:',
            '只有Alice能够计算出S_A().这个要求也是Alice要对S_A保密的原因：如果她不能做到这一点,就会失去她的唯一特性')
        print('  Bob取得Alice的公钥P_A')
        print('  Bod计算出响应于M的密文C=P_A(M),并把C发送给Alice')
        print('  当Alice收到密文C后,运用自己的密钥S_A恢复出原始信息：M=S_A(C)')
        print('类似地,在公钥系统中可以很容易地实现数字签名.假设现在Alice希望把一个数字签署的回应M发送给Bob.数')
        print('  Alice运用她的密钥S_A计算出信息M‘的数字签名o=S_A(M\')')
        print('  Alice把该消息/签名对(M\', o)发送给Bob')
        print('  当Bob收到(M\', o)时,他可以利用Alice的公钥通过验证等式M\'=P_A(o)来证实该消息的确是Alice发出的')
        print('因为数字签名有一条重要的性质,就是它可以被任何能取得签署者的公钥的人所验证.',
            '一条签署过的信息可以被一方确认过后再传送到其他地方,他们也同样能对该签名进行验证',
            '例如这条消息可能是Alice发给Bob的一张电子支票')
        print('RSA加密系统')
        print('  在RSA公钥加密系统中,一个参加者按下列过程来创建他的公钥与密钥')
        print('  1.随机选取两个大素数p和q,且p!=q,例如素数p和q可能各有512位(二进制)')
        print('  2.根据式n=pq计算出n的值')
        print('  3.选取一个与phi(n)互质的小奇数e,其中phi(n)=(p - 1)(q - 1)')
        print('  4.对模phi(n),计算出e的乘法逆元d的值')
        print('  5.输出对P=(e,n),把它作为RSA公钥')
        print('  6.把对S=(d,n)保密,并把它作为RSA密钥')
        print('运用MODULAR-EXPONENTIATION,来实现上述公钥与密钥的有关操作.为了分析这些操作的运行时间,假定公钥(e,n)和密钥(d,n)满足lge=O(1),',
            'lgd<=b以及lgn<=b,以及lgn<=b.则应用公钥操作,需要执行O(1)次模乘法运算和O(b^2)次位操作',
            '应用密钥操作,需要执行O(b)次模乘法运算和O(b^3)次位操作')
        print('使RSA与一个公开的单向散列函数h相结合,其中的单向三列函数是易于计算的,但是对这种函数来说,要找出两条消息M和M\'满足：h(M)和h(M\')',
            '这在计算上是不可行的.h(M)的值消息M的一个短的\"指纹\".')
        print('练习31.7-1 考察一个RSA密钥集合,其中p=11,q=29,n=319,e=3.在密钥中用到的d值应当是多少.对消息M=100加密后得到什么消息')
        print('练习31.7-2 证明：如果Alice的公开指数e等于3,并且对方获得了Alice的秘密指数d',
            '则对方能够在关于n的位数的多项式时间内对Alice的模n进行分解')
        print('练习31.7-3 证明：在如下意义中上说,RSA是乘法的：P_A(M1)P_A(M2)=P_A(M1M2)(mod n)')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_8:
    """
    chapter31.8 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.8 note

        Example
        ====
        ```python
        Chapter31_8().note()
        ```
        """
        print('chapter31.8 note as follow')
        print('31.8 素数的测试')
        print('寻找大素数的问题，首先讨论素数的密度，接着讨论一种似乎可行的(但不完全)测试素数的方法,',
            '然后,介绍一种由Miller和Rabin发现的有效的随机素数测试算法')
        print('素数的密度')
        print('  在很多应用领域(如密码学中),需要找出大的“随机”素数,幸运的是,大素数并不算太少,因此测试适当的随机整数',
            '直至找到素数的过程也不是太费时的.素数分布函数pi(n)秒睡了小于或等于n的素数的数目,例如pi(10)=4',
            '因为小于或等于10的素数有4个,分别为2,3,5,7')
        print('定理31.37 (素数定理) lim(pi(n)/(n / lnn)) = 1')
        print('对于较小的n,近似计算式n/lnn可以给出pi(n)相当精确的估计值。例如当n=10^9时,其误差不超过6%,这时pi(n)=50 847 534,且n/lnn=48 254 942',
            '对于研究数论的人来说,10^9是一个小数字')
        print('运用素数定理,可以估计出一个随机选取的整数n是素数的概率为1/lnn.因此,为了找出一个长度与n相同的素数,大约要检查在n附近随机选取的lnn个整数',
            '例如为了找出一个512位长的素数，大约需要对ln 2^512=335个随机选取的512位长的整数进行素数测试.(通过只选择奇数,就可以把这个数字减少一半)')
        print('伪素数测试过程')
        print('  考察一种“几乎可行”的素数测试方法,事实上,对很多实际应用来说,这种方法已经是相当好的方法了.后面还将对这个方法作精心的改进',
            '以消除其中存在的小的缺陷.设Zn+表示Zn中的非零元素：Zn+={1,2,...,n-1}')
        print('  如果n是素数,则Zn+=Zn*；如果n是一个合数,而且a^(n-1)=1(mod n)')
        print('  则说n是一个基为a的伪素数。费马定理蕴含着如果n是一个素数,则对Zn+中的每一个a,n满足等式。因此,如果能找出任意的a∈Zn+',
            '使得n不满足等式,那么n当然就是合数。这个命题的逆命题也几乎成立。因此，这一衡量标准几乎是素数测试的正确标准')
        print('  对给定规模我的基于2的伪素数的数目能做出更加精确地估计,就可以得到被上述过程称为素数的一个随机选取的512位数,',
            '是基于2的伪素数的概率不到1/10^20,而被上述过程称为素数的一个随机选取的1024位数,是基于2的伪素数的概率不到1/10^41',
            '因此,如果子式试着为某个应用找到一个大的素数,通过随机选取大的数字,在所有实际用途中几乎会永远不会出错')
        print('  但是当测试素数的数字不是随机选取的时候,就需要一个更好的方法来进行素数测试')
        print('Carmochael数')
        print('  前三个Carmichael数是561,1105,1729.在小于100000000的数中,只有255个Carmichael数')
        print('Miller-Rabin随机性素数测试方法')
        print('  Miller-Rabin素数测试方法对简单测试过程PSEUDOPRIME做了两点改进,从而解决了其中存在的问题')
        print('它试验了数个随机选取的基值a,而不是仅仅试验一个基值')
        print('当计算每个模取幂的值时，注意在最后一组平方里是否发现了对模n来说1的非平凡平方根。如果发现这样的根存在,终止执行并输出结果COMPOSITE')
        print('如果n是一个b位数,则MILLER-RABIN需要执行O(sb)次算术运算和O(sb^3)次位操作,这是因为从渐进意义上说,需要执行的工作仅是s次取模幂运算')
        print('Miller-Rabin素数测试的出错率')
        print('  如果Miller-Rabin输出Prime,则它仍有一种很小的可能会产生错误.但是,与PSEUDOPRIME不同的是,出错的可能性并不依赖于n;对该过程也不存在坏的输入',
            '相反地,它取决于s的大小和在选取基值a时抽签的运气.同时,由于每次测试比对作简单的检查更严格,因此从总的原则上,对随机选取的整数n,其出错率应该是很小的')
        print('定理31.38 如果n是一个奇合数,则n为合数的证据的数目至少为(n-1)/2')
        print('定理31.39 对任意奇数n>2和正整数s,Miller-Rabin(n, s)出错的概率至多为2^(-s)')
        print('  因此,对可以想象得到的几乎所有的应用,选取s=50应该是足够的')
        print('练习31.8-1 证明：如果一个奇整数n>1不是素数或素数的幂,则对模n存在一个1的非平凡方根')
        print('练习31.8-2 可以稍稍把欧拉定理加强为以下形式,对所有a∈Zn,证明la(n) | phi(n),如果la(n) |n - 1,则合数为Carmichael',
            '最小的Carmichael数为561 = 3 * 11 * 17')
        print('练习31.8-3 证明：如果对模n,x是1的非平凡平方根,则gcd(x - 1, n)和gcd(x + 1,n)都是n的非平凡约数')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_9:
    """
    chapter31.9 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.9 note

        Example
        ====
        ```python
        Chapter31_9().note()
        ```
        """
        print('chapter31.9 note as follow')
        print('31.9 整数的因子分解')
        print('假设希望将一个整数n分解为素数的积.通过上面一节所讨论的素数测试,可以知道n是否是合数,但如果n是合数,它并不能指出n的素数因子',
            '对一个大整数n进行因子分解,似乎要比仅确定n是素数还是合数')
        print('即使用当今的超级计算机和现行的最佳算法,要对一个1024位的数进行因子分解也还是不可行的')
        print('Pollard的rho启发式方法')
        print('  对到B的所有整数进行试除,可以保证完全获得到B^2的任意数的因子分解.下列过程做相同的工作量,就能对到B^4的任意数进行因子分解(除非运气不佳)',
            '由于该过程仅仅是一种启发性方法,因此既不能保证其运行时间也不能保证其运行成功,不过在实际应用中该过程还是非常有效的.',
            'POLLARD-RHO过程的另一个优点是,它只使用固定量的存储空间')
        print('  POLLARD-RHO在while循环执行大约Θ(sqrt(p))次迭代后,会输出n的一个因子p.因此,如果n是合数,则大约经过n^(1/4)次更新操作后',
            '可以预计该过程已经找到了要把n完全分解因子所需要的足够的约数,这是由于除了可能有最大的一个素因子外,n的每一个素因子p均小于sqrt(n)')
        print('  根据生日悖论，在序列出现回路之前预计要执行的步数为Θ(sqrt(n))')
        print('练习31.9-1 在图31-7a所示的执行过程中,过程POLLARD-RHO在何时输出1387的因子73')
        print('练习31.9-2 假设已知函数f:Zn->Zn和一个初值x0∈Zn.定义xi=f(xi-1),i=1,2,...,设t和u>0是满足x(t+i)=x(t+u+i),i=0,1,...的最小值',
            '在Pollard的rho算法的术语中')
        print('练习31.9-3 要发现形如p^e的数(其中p是素数,e>1)的一个因子,POLLARD-RHO要执行多少步')
        print('练习31.9-4 POLLARD-RHO的一个缺陷是在其递归过程的每一步,都要计算一个gcd.','有人建议gcd的计算进行批处理：累计一行中数个连续的xi的积',
            '然后在gcd计算中使用该积而不是xi.请说明如何实现这一设计思想,以及为什么它是正确的,在处理一个b位数n时,所选取的最有效的批处理规模是多大')
        print('思考题33-1 二进制的gcd算法')
        print('  在大多数计算机上,与计算余数的执行速度相比,减法运算、测试一个二进制整数的奇偶性运算以及折半运算的执行速度都要更快些')
        print('  a) 证明：如果a和b都是偶数,则gcd(a,b)=2gcd(a / 2, b / 2)')
        print('  b) 证明：如果a是奇数,b是偶数,则gcd(a,b)=gcd(a, b / 2)')
        print('  c) 证明：如果a和b都是奇数,则gcd(a, b)=gcd((a-b)/2, b)')
        print('  d) 设计一个有效的二进制gcd算法,输入为整数a和b(a>=b)并且算法的运行时间为O(lga).',
            '假定每个减法运算、测试奇偶性运算以及折半运算都能在单位时间内执行')
        print('思考题33-2 对欧几里得算法中位操作的分析')
        print('  a) 证明：用普通的“纸和笔”算法来进行长除法运算：用a除以b,得到商q和余数r,需要执行O((1+lgq)lgb)次位操作')
        print('  b) 定义u(a, b)=(1+lga)(1+lgb).证明：过程EUCLID在把计算gcd(a,b)的问题转化为计算gcd(b,a mod b)的问题时',
            '所执行的位操作次数至多为c(u(a, b))-u(b, a mod b),其中c>0为某一个足够大的常数')
        print('  c) 证明：在一般情况下EUCLID(a, b)需要执行的位操作次数为O(u())')
        print('思考题33-3 关于斐波那契数的三个算法')
        print('  a) 在已知n的情况下,本问题对计算第n个斐波那契数Fn的三种算法的效率进行了比较,假定两个数的加法、减法和乘法的代价都是O(1),与数的大小无关')
        print('  b) 试说明如何运用记忆法在O(n)的时间内计算Fn')
        print('  c) 试说明如何仅用整数加法和乘法运算,就可以在O(lgn)的时间内计算Fn')
        print('  d) 现在假设对两个b位数相加需要Θ(b)的时间,对两个b位数相乘需要Θ(b^2)的时间',
            '如果这样来更合理地估计基本算术运算的代价,则这三种方法的运行时间又是多少')
        print('思考题33-4 二次余数')
        print('  设p是一个奇素数,如果关于未知量x的方程x^2=a (mod p)有解,则数a∈Zp就是一个二次余数')
        print('  a) 证明：对模p,恰有(p-1)/2个二次余数')
        print('  b) 如果p是素数,对a∈Zp,定义勒让德符号(a/p)等于1,如果a是对模p的二次余数;否则其值等于-1.证明:如果a∈Zp,则(a/p)=a^(p-1)/2 (mod p)')
        print('试写出一个有效的算法,使其能确定一个给定的数a是否对模p的二次余数.分析所给算法的效率')
        print('  c) 证明：如果p是形如4k+3的素数,且a是Zp中一个二次余数,则a^(k+1) mod p是对模p的a的平方根',
            '找出一个对模p的二次余数a的平方根需要多长时间')
        print('  d) 试描述一个有效的随机算法,来找出一个以任意素数p为模的非二次余数,以及Zp中不是二次余数的成员',
            '所给出的算法平均需要执行多少次算术运算')
        print('')
        print('')
        print('')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

chapter31_1 = Chapter31_1()
chapter31_2 = Chapter31_2()
chapter31_3 = Chapter31_3()
chapter31_4 = Chapter31_4()
chapter31_5 = Chapter31_5()
chapter31_6 = Chapter31_6()
chapter31_7 = Chapter31_7()
chapter31_8 = Chapter31_8()
chapter31_9 = Chapter31_9()

def printchapter31note():
    """
    print chapter31 note.
    """
    print('Run main : single chapter thirty-one!')
    chapter31_1.note()
    chapter31_2.note()
    chapter31_3.note()
    chapter31_4.note()
    chapter31_5.note()
    chapter31_6.note()
    chapter31_7.note()
    chapter31_8.note()
    chapter31_9.note()

# python src/chapter31/chapter31note.py
# python3 src/chapter31/chapter31note.py

if __name__ == '__main__':  
    printchapter31note()
else:
    pass

```

```py

import random as _rand

class _NumberTheory:
    """
    数论相关算法集合
    """
    def __init__(self):
        """
        数论相关算法集合
        """
        pass

    def gcd(self, a : int, b : int):
        """
        Summary
        ====
        求两个数的最大公约数

        Args
        ===
        `a`: 数字1

        `b`: 数字2

        Return
        ===
        `num` : 最大公约数

        Example
        ===
        ```python
        >>> gcd(24, 30)
        >>> 6
        ```

        """
        assert a >= 0 and b >= 0
        if a == 0 and b == 0:
            return 0
        return self.euclid(a, b)

    def euclid(self, a, b):
        """
        欧几里得算法
        """
        if b == 0:
            return a
        return self.euclid(b, a % b)

    def extend_euclid(self, a, b):
        """
        推广欧几里得算法
        """
        if b == 0:
            return (a, 1, 0)
        (d_ , x_, y_) = self.extend_euclid(b, a % b)
        d, x, y = d_, y_, x_ - (a // b) * y_
        return (d, x, y)

    def ismutualprime(self, a : int, b : int):
        """
        判断两个数是不是互质数
        Args
        ===
        `a`: 数字1

        `b`: 数字2
        """
        return self.gcd(a, b) == 1

    def modular_linear_equation_solver(self, a, b, n):
        """
        求模线性方程组
        """ 
        d, x, y = self.extend_euclid(a, n)
        if d or b:
            x0 = x * (b / d) % n
            for i in range(d):
                print((x0 + i * (n / d)) % n)
        else:
            print('no solotion')

    def modular_exponentiation(self, a, b, n):
        """
        运用反复平方法求数的幂
        """
        c = 0
        d = 1
        bit = bin(b)
        bit = bit[2::]
        bit_list = [int(c) for c in bit]
        d_list = []
        for b in bit_list:
            c = 2 * c
            d = (d * d) % n
            if b == 1:
                c += 1
                d = (d * a) % n
        return d
    
    def witness(self, a, n):
        """
        WIRNESS测试函数
        """
        bit_str = bin(n - 1)
        t = 0
        length = len(bit_str)
        for i in range(length - 1, -1, -1):
            if bit_str[i] == '0':
                t += 1
            else:
                break
        bit_str = bit_str[0:length - t]
        u = int(bit_str, 2)
        x = [0] * (t + 1)
        x[0] = self.modular_exponentiation(a, u, n)
        for i in range(1, t + 1):
            x[i] = (x[i - 1] ** 2) % n
            if x[i] == 1 and x[i - 1] != 1 and x[i - 1] != (n - 1):
                return True
        if x[t] != 1:
            return True
        return False

    def miller_rabin(self, n, s):
        """
        Miller-Rabin随机性素数测试方法
        """
        for j in range(1, s + 1):
            a = _rand.randint(1, n - 1)
            if self.witness(a, n):
                return "Composite"
        return "Prime"

    def pollard_rho(self, n):
        """
        整数的因子分解 Pollard的rho启发式方法

        Args
        ===
        `n` : 被分解的数字

        """
        i = 1
        x = _rand.randint(0, n - 1)
        y = x
        k = 2
        while True:
            i += 1
            x = (x ** 2 - 1) % n
            d = self.gcd(y - x, n)
            if d != 1 and d != n:
                print(d)
            if i == k:
                y = x
                k = 2 * k

__number_theory_instance = _NumberTheory()

gcd = __number_theory_instance.gcd
euclid = __number_theory_instance.euclid
extend_euclid = __number_theory_instance.extend_euclid
ismutualprime = __number_theory_instance.ismutualprime
modular_linear_equation_solver = __number_theory_instance.modular_linear_equation_solver
modular_exponentiation = __number_theory_instance.modular_exponentiation
miller_rabin = __number_theory_instance.miller_rabin
pollard_rho = __number_theory_instance.pollard_rho

def test():
    """
    测试函数
    """
    print(gcd(24, 30))
    print(euclid(24, 30))
    print(extend_euclid(24, 30))
    print(gcd(24, 30))
    print(modular_linear_equation_solver(14, 30, 100))
    print(modular_exponentiation(7, 560, 561))
    print(miller_rabin(561, 10))
    print(pollard_rho(12))

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter32/chapter32note.py
# python3 src/chapter32/chapter32note.py
"""

Class Chapter32_1

Class Chapter32_2

Class Chapter32_3

Class Chapter32_4

"""
from __future__ import absolute_import, division, print_function

import math
import re
import numpy as np

if __name__ == '__main__':
    import stringmatch as sm
else:
    from . import stringmatch as sm

class Chapter32_1:
    """
    chapter32.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.1 note

        Example
        ====
        ```python
        Chapter32_1().note()
        ```
        """
        print('chapter32.1 note as follow')
        print('第32章 字符串匹配')
        print('在文本编辑程序中,经常出现要在一段文本中找出某个模式的全部出现位置这一问题。典型情况是,一段文本是正在编辑的文件,',
            '所搜寻的模式是用户提供的一个特定单词。解决这个问题的有效算法能极大地提高文本编辑程序的响应性能',
            '字符串匹配算法也常常用于其他方面,例如在DNA序列中搜寻特定的模式')
        print('字符串匹配问题的形式定义是这样的:假设文本是一个长度为n的数组T[1..n],模式是一个长度为m<=n的数组P[1..m].',
            '进一步假设P和T的元素都是属于有限字母表∑表中的字符.例如可以有∑={0,1}或∑={a,b,...,z},字符数组P和T常称为字符串')
        print('如果0<=s<=n-m,并且T[S+1,...,s+m]=P[1..m](即对1<=j<=m,有T[s+j]=P[j]),则说模式P在文本T中出现且位移为s.',
            '(或者等价地,模式P在文本T中从位置s+1开始出现)。如果P在T中出现且位移为s,则称s为一个有效位移,否则称s为无效位移',
            '这样一来,字符串匹配问题就变成一个在一段指定的文本T中,找出某指定模式P出现所有有效位移的问题')
        print('本章的每个字符串匹配算法都对模式进行了一些预处理,然后找寻所有有效位移;我们称第二步为“匹配”.',
            '每个算法的总运行时间为预处理和匹配时间的总和.')
        print('32.2节介绍由Rabin和Karp发现的一种有趣的字符串匹配算法,该算法在最坏情况下的运行时间为Θ((n-m+1)m),虽然这一时间并不比朴素的算法好',
            '但是在平均情况和实际情况中,该算法的效果要好的多.这种算法也可以很好地推广到解决其他的模式匹配问题')
        print('32.3节中描述另一种字符串匹配算法,该算法构造一个特别设计的有限自动机,用来搜寻某给定模式P在文本中的出现的位置',
            '此算法用O(m|∑|)的预处理时间,但只用Θ(n)的匹配时间')
        print('32.4节介绍与其类似但更巧妙的Knuth-Morris-Pratt(或KMP)算法。该算法的匹配时间同样为Θ(n),但是将预处理时间降至Θ(m)')
        print('算法          预处理时间        匹配时间')
        print('朴素算法          0           O((n-m+1)m)')
        print('Rabin-Karp       Θ(m)        O((n-m+1)m)')
        print('有限自动机算法   O(m|∑|)         Θ(n)')
        print('KMP算法          Θ(m)           Θ(n)')
        print('记号与术语')
        print('  用∑*表示用字母表∑中的字符形成的所有有限长度的字符串的集合.在本章中仅考虑长度有限的字符串',
            '长度为0的空字符串用e表示,它也属于∑*.字符串x的长度用|x|表示.两个字符串x和y的连接表示为xy,其长度为|x|+|y|,由x的字符接y的字符组成')
        print('  如果对某个字符串与y∈∑*,有x=wy,就说字符串w是字符串x的前缀,表示为w∝x,得知|w|<=|x|.空字符串e既是每个字符串的前缀,也是每个字符串的后缀',
            '例如,有ab>abcca,cca<abcca.对任意字符串x和y及任意字符a,x>y当且仅当xa>ya.注意>和<都是传递关系')
        print('引理32.1(重叠后缀定理)假设x,y和z是满足x>z和y<z的三个字符串.如果|x|<=|y|,则x>y;如果|x|>=|y|,则y>x;如果|x|=|y|,则x=y')
        print('  本章中允许把比较两个等长的字符串是否相等的操作当做原语操作.如果对字符串的比较是从左往右进行,并且发现一个不匹配字符时比较就终止,',
            '则假设这样一个测试过程所需的时间是关于所发现的匹配字符数目的线性函数')
        print('32.1 朴素的字符串匹配算法')
        print('朴素的字符串匹配算法:它用一个循环来找出所有有效位移,该循环对n-m+1可能的每一个s值检查条件P[1..m]=T[s+1..s+m]')
        print('这种朴素的字符串匹配过程可以形象地看成用一个包含模式的“模板”沿文本滑动,同时对每个位移注意模板上的字符是否与文本的相应字符相等')
        print('NATIVE-STRING-MATCHER的运行时间为Θ((m-m+1)m)')
        print('在本章中还要介绍一种算法,它的最坏情况预处理时间为Θ(m),最坏情况匹配时间为Θ(n)')
        print('练习31.1-1 解答过程如下：')
        P = '0001'
        T = '000010001010001'
        sm.native_string_matcher(T, P)
        print('练习31.1-2 假设模式P中的所有字符都是不同的。试说明如何对一段n个字符的文本T加速过程NATIVE-STRING-MATCHER的执行速度,',
            '使其运行时间达到O(n)')
        print('练习31.1-3 假设模式P和文本T是长度分别为m和n的随机选取的字符串,其字符串属于d个元素的字母表∑={0,1,...,d-1},其中d>=2',
            '证明朴素算法第4行中隐含的循环所执行的字符比较的预计次数为')
        print('   (n-m+1)(1-d**-m)/(1-d**-1)<=2(n-m+1)')
        print('练习31.1-4 假设允许模式P中包含一个间隔字符◇,该字符可以与任意的字符串匹配(甚至可以与长度为0的字符串匹配)',
            '例如,模式ab◇ba◇c')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_2:
    """
    chapter32.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.2 note

        Example
        ====
        ```python
        Chapter32_2().note()
        ```
        """
        print('chapter32.2 note as follow')
        print('Rabin-Karp算法')
        print('在实际应用中,Rabin和Karp所建议的字符串匹配算法能够较好的运行,我们还可以从中归纳出有关问题的其他算法,如二维模式匹配')
        print('Rabin-Karp算法预处理时间为')
        print('Θ(m),在最坏情况下的运行时间为O((n-m+1)m),但是它的平均情况运行时间还是比较好的')
        print('假定∑={0,1,2,...,9},这样每个字符都是一个十进制数字(一般情况下,可以假定每个字符都是基数为d的表示法中一个数字,d=|∑|).',
            '可以用一个长度为k的十进制数来表示由k个连续字符组成的字符串.因此,字符串31415就对应于十进制数31415',
            '如果输入字符既可以看做图形符号,也可以看做数字')
        print('已知一个模式P[1..m],设p表示其相应的十进制数的值。对于给定的文本T[1..n],用ts来表示其长度为m的子字符串T[s+1..s+m](s=0,1,...,n - m)相应的十进制数的值')
        print('当然,用ts来表示其长度为m的子字符串T[s+1..s+m](s=0,1,..,n-m)相应十进制数的值.当然,ts=p当且仅当T[s+1..s+m]=P[1..m],因此s是有效位移当且仅当ts=p',
            '如果能够在Θ(m)的时间内计算出p的值,并在总共Θ(n-m+1)的时间内计算出所有ts的值,那么通过把p值与每个ts值进行比较,能够在Θ(n)的时间内,求出有效位移s')
        print('可以运用霍纳法则,在Θ(m)的时间内计算出p的值：')
        print('  p=P[m]+10(P[m-1]+10(P[m-2]+...+10(P[2]+10P[1])...))')
        print('类似地,也可以在Θ(m)的时间内,根据T[1..m]计算出t0的值')
        print('为了在Θ(n-m)的时间内计算出剩余的值t1,t2,...,tn-m,可以在常数时间内根据ts计算出ts+1,这是因为霍纳法则')
        print('RABIN-KARP-MATCHER的预处理时间Θ(m),其匹配时间在最坏情况下为Θ((n-m+1)m),因为Rabin-Karp算法与朴素的字符串匹配算法一样,对每个有效位移进行显示验证',
            '如果P=a^m并且T=a^n,则验证所需的时间为Θ((n-m+1)m),因为n-m+1可能的位移中每一个都是有效位移')
        print('在许多实际作用中,有效位移数很少(如只有常数c个),因此,算法的期望匹配时间为O((n-m+1)+cm)=O(n+m),',
            '再加上处理伪命中点所需的时间.假设减少模q的值就像是从∑*到Zq上的一个随机映射,基于这种假设,进行启发性分析',
            '要正式证明这个假设是比较困难的,但是有一种可行的方法,就是假定q是从适当大的整数中随机得出的.可以预计伪命中的次数为O(n/q)',
            '因为可以估计出任意的ts对模q等价于p的概率为1/q.')
        print('Rabin-Karp算法的期望运行时间为:O(n)+O(m(v+n/q))')
        print('练习32.2-1 如果取模q=11,那么当Rabin-Karp匹配算法在文本T=3141592653589793中与搜寻模式P=26,会遇到多少个伪命中点')
        sm.native_string_matcher('3141592653589793', '26')
        sm.rabin_karp_matcher('3141592653589793', '26', 10, 11)
        print('练习32.2-2 如何扩展Rabin-Karp方法,使其能解决这样的问题:如何在文本字符串中搜寻出给定的k个模式中任何一个出现',
            '起初假定所有k个模式都是等长的.然后扩展算法允许不同长度的模式')
        print('练习32.2-3 试说明如何扩展Rabin-Karp方法以处理下列问题,在一个n*n二维字符串中搜寻出给定的m*m模式',
            '(可以使该模式在水平方向和垂直方向移动,但不可以把模式旋转)')
        print('练习32.2-4 Alice有一份很长的n位文件的复印件A=<an-1,an-2,...,a0>,Bob也有一份类似的文件B=<bn-1,bn-2,...,b0>',
            'Alice和Bob都希望知道他们的文件是否一样，为了避免传送整个文件A或B.',
            '运用下列快速概率检查手段,一起选择一个素数q>1000n,并从{0,1,...,q-1}中随机选取一个整数x,然后Alice求出：',
            'A(x)=(∑aixi) mod q)的值,Bob也用类似的方法计算出B(x).',
            '证明：如果A≠B,则A(x)=B(x)的概率至多为1/1000;如果两个文件相同,则A(x)的值必定等于B(x)的值') 
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_3:
    """
    chapter32.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.3 note

        Example
        ====
        ```python
        Chapter32_3().note()
        ```
        """
        print('chapter32.3 note as follow')
        print('32.3 利用有限自动机进行字符串匹配')
        print('很多字符串匹配算法都要建立一个有限自动机,它通过对文本字符串T进行扫描的方法,',
            '找出模式P的所有出现位置.建立这样自动机的方法,用于字符串匹配：它们只对每个文本字符检查一次,并且检查每个文本字符的时间为常数',
            '因此,在建立好自动机后所需要的时间为Θ(n),但是如果∑很大,建立自动机所花的时间也可能是很多的')
        print('在本节的开头先定义有限自动机概念.考察一种一种特殊的字符串匹配自动机,并说明如何利用它找出一个模式在文本中的出现位置',
            '包括对一段给定的文本,如何模拟出字符串匹配自动机的执行步骤的一些细节.',
            '将说明对一个给定的输入模式,如何构造相应的字符串匹配自动机')
        print('有限自动机')
        print('  一个有限自动机M是一个5元组(Q,q0,A,∑,d)')
        print('   Q是一个状态的有限集合')
        print('   q0∈Q是初始状态')
        print('   A∈Q是一个接受状态集合')
        print('   ∑是有限的输入字母表')
        print('   d是一个从Q×∑到Q的函数,称为M的转移函数')
        print('  有限自动机开始于状态q0,每次读入输入字符串的一个字符.如果有限自动机在状态q时读入了输入字符a,则它从状态q变为状态d(q,a)(进行了一次转移).',
            '每当其状态q属于A时,就说自动机M接受了所有读入的字符串。没有被接收的输入称为被拒绝的输入')
        print('  有限自动机M可以推导出一个函数∮,称为终止函数,它是从∑*到Q的函数,并满足:∮(w)是M在扫描字符串w终止时的状态.',
            '因此,M接受字符串w当且仅当∮(w)∈A,函数∮由下列递归关系定义∮(e)∈q0',
            '∮(wa)=d(∮(w), a) 对于w∈∑*, a∈∑')
        print('字符串匹配自动机')
        print('  对每个模式P都存在一个字符串匹配自动机,必须在预处理阶段,根据模式构造出相应的自动机后,才能利用它来搜寻文本字符串',
            '关于模式P=ababaca的有限自动机的构造过程。从现在开始,假定P是一个已知的固定模式.为了使说明上的简洁,在下面的概念中将不特别指出对P的依赖关系')
        print('  为了详细说明与给定模式P[1..m]相应的字符串匹配自动机,首先定义一个辅助函数a,称为相应P的后缀函数。',
            '函数a是一个从∑*到{0,1,...,m}上定义的映射,a(x)是x的后缀P的最长前缀的长度：a(x)=max{k:Pk>x}')
        print('  为了清楚地说明字符串匹配自动机的操作过程,给出一个简单而有效的程序,用来模拟这样一个自动机(用它的变迁函数d来表示),在输入文本T[1..n]中,',
            '寻找长度为m的模式P的出现位置的过程,对于长度为m的模式的任意字符串匹配自动机来说,状态Q为{0,1,...,m},初始状态为0,唯一的接收态是状态m')
        print('  由FINITE-AUTOMATON-MATCHER的简单循环结构可以看出,对于一个长度为n的文本字符串,它的匹配时间为Θ(n)',
            '但是,这一匹配时间没有包括计算变迁函数d所需要的预处理时间.将在证明FINITE-AUTOMATON-MATCHER的正确性以后,再来讨论这一问题')
        print('  考察自动机在输入文本T[1..n]上进行的操作.将证明自动机扫过字符T[i]后,其状态为d(Ti).因为d(Ti)=m当且仅当P>Ti,',
            '所以自动机处于接收状态m,当且仅当模式P已经被扫描过,为了证明这个结论,要用到下面两条关于后缀函数o的引理')
        print('引理32.2 (后缀函数不等式) 对任意字符串x和字符a,有o(xa)>=o(x)+1')
        print('引理32.3 (后缀函数递归引理) 对任意x和字符a,如果q=o(x),则o(xa)=o(Pqa)')
        print('定理32.4 如果∮是字符串匹配自动机关于给定模式P的终态函数,T[1..n]是自动机的输入文本,对i=0,1,...,n,有∮(Ti)=o(Ti)')
        print('计算变迁函数')
        print('练习32.3-1 对模式P=aabab构造出相应的字符串匹配自动机,并说明它在文本字符串T=aaababaabaababaab上的操作过程')
        print(sm.finite_automaton_matcher('aaababaabaababaab', 10, 8))
        print('练习32.3-2 对字母表∑={a, b},画出与模式ababbabbababbababbabb相应的字符串匹配自动机状态转换图')
        print('练习32.3-3 如果由Pk>Pq蕴含着k=0或k=q,则称模式P是不可重叠的.试描述与不可重叠模式相应的字符串匹配自动机的状态转换图')
        print('练习32.3-4 已知两个模式P和P`,试描述如何构造一个有限自动机,使之能确定其中任意一个模式的所有出现位置.要求尽量使自动机的状态数最小')
        print('练习32.3-5 已知一个包括间隔字符的某模式P,说明如何构造一个有限自动机,使其在O(n)的时间内,找出P在文本T中的一次出现位置,其中n=|T|')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_4:
    """
    chapter32.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.4 note

        Example
        ====
        ```python
        Chapter32_4().note()
        ```
        """
        print('chapter32.4 note as follow')
        print('32.4 Knuth-Morris-Pratt算法')
        print('Knuth、Morris和Pratt三人设计的线性时间字符串匹配算法。这个算法不用计算变迁函数d,匹配时间为Θ(n),只要用到辅助函数pi[1,m]',
            '它是在Θ(m)时间内,根据模式预先计算出来的.数组pi使得我们可以按需要,“现场”有效地计算(在平摊意义上来说)变迁函数d.',
            '粗略地说,对任意状态q=0,1,...,m和任意字符a∈∑,pi[q]的值包含了与a无关但在计算d(q,a)时需要的信息',
            '由于数组pi只有m个元素,而d有Θ(m|∑|个值,所以通过预先计算pi而不是d,使得时间减少了一个|∑|因子)')
        print('关于模式的前缀函数')
        print('  模式的前缀函数pi包含有模式与其自身的位移进行匹配的信息.这些信息可用于避免在朴素的字符串匹配算法中,对无位移进行测试,',
            '也可以避免在字符串匹配自动机中,对d的预先计算过程')
        print('KMP-MATCHER的大部分过程都是在模仿FINITE-AUTOMATON-MATCHER.KMP-MATCHER调用了一个辅助过程COMPUTE-PREFIX-FUNCTION来计算pi')
        print('运行时间分析')
        print('  运用平摊分析方法进行分析后可知,过程COMPUTE-PREFIX-FUNCTION的运行时间为Θ(m)')
        print('  在类似的平摊分析中,如果用q的值作为势函数,则KMP-MATCHER的匹配时间为Θ(n)',
            '与FINITE-AUTOMATON-MATCHER相比,通过运用pi而不是d,可使对模式进行预处理所需的时间由O(m|∑|)下降为Θ(m),同时保持实际的匹配时间为Θ(n)')
        print('前缀函数计算的正确性')
        print('  通过对前缀函数pi进行迭代,就能够列举出是某给定前缀Pq的后缀的所有前缀Pk,设',
            'pi*[q]={pi[q],pi(2)[q],pi(3)[q],...,pi(t)[q]}')
        print('引理32.5 (前缀函数迭代定理) 设P是长度为m的模式,其前缀函数为pi,对q=1,2,...,m,有pi*[q]={k:k<q且Pk>Pq}')
        print('引理32.6 设P是长度为m的模式,pi是P的前缀函数.对q=1,2,...,m,如果pi[q]>0,则pi[q]-1∈pi*[q-1]')
        print('推论32.7 设P是长度为m的模式,pi是P的前缀函数,对q=2,3,...,m')
        print('KMP算法的正确性')
        print('  过程KMP-MATCHER可以看做是过程FINITE-AUTOMATON-MATCHER的一次重新实现')
        print('练习32.4-1 当字母表为∑={a,b},计算相应于模式ababbabbabbababbabb的前缀函数pi')
        print('练习32.4-2 给出关于q的函数pi*[q]的规模的上界.举例说明所给出的上界是严格的')
        print('练习32.4-3 试说明如何通过检查字符串PT的pi函数,来确定模式P在文本T中的出现位置(由P和T并置形成的长度为m+n的字符串)')
        print('练习32.4-4 试说明如何通过以下方式对过程KMP-MATCHER进行改进:把第7行(不是第12行中)出现的pi替换为pi‘.对q=1,2,...,m的递归定义如下：')
        print('练习32.4-5 写出一个线性时间的算法,','以确定文本T是否是另一个字符串T‘的循环旋转,例如arc和car是彼此的循环旋转')
        print('练习32.4-6 给出一个有效的算法,计算出相应于某给定模式P的字符串匹配自动机的变迁函数d',
            '所给出的算法的运行时间应该是O(m|∑|).(提示：证明:如果q=m或P[q+1]!=a,则d(q,a)=d(pi[q],a))')
        print('思考题32-1 基于重复因子的字符串匹配')
        print('  设yi表示字符串y与其自身并置i次所得的结果.例如(ab)^3=ababab.如果对某个字符串y∈∑*和某个r>0有x=y^r,则称字符串x∈∑*具有重复因子r.',
            '设p(x)表示满足x具有重复因子r的最大值')
        print('  (a) 写出一个有效算法以计算出p(Pi)(i=1,2,...,m),算法的输入为模式P[1..m].算法的运行时间是多少？')
        print('  (b) 对任何模式p[1..m],设p(P)定义为max(1<=i<=m)p(Pi).证明：如果从长度为m的所有二进制字符串所组成的集中随机地选择模式P,则p*(P)的期望值是O(1)')
        print('  (c) 论证下列字符串匹配算法可以在O(p*(P)n+m)的运行时间内,正确地找出模式P在文本T[1..n]中的所有出现位置')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

chapter32_1 = Chapter32_1()
chapter32_2 = Chapter32_2()
chapter32_3 = Chapter32_3()
chapter32_4 = Chapter32_4()

def printchapter32note():
    """
    print chapter32 note.
    """
    print('Run main : single chapter thirty-two!')
    chapter32_1.note()
    chapter32_2.note()
    chapter32_3.note()
    chapter32_4.note()

# python src/chapter32/chapter32note.py
# python3 src/chapter32/chapter32note.py

if __name__ == '__main__':  
    printchapter32note()
else:
    pass

```

```py

import random as _rand

class _StringMatch:
    """
    字符串匹配相关算法
    """
    def __init__(self):
        """
        字符串匹配相关算法
        """
        pass

    def native_string_matcher(self, T : str, P : str):
        """
        朴素字符串匹配
        Args
        ===
        `T` : str

        `P` : str
        """
        n = len(T)
        m = len(P)
        if n < m:
            self.native_string_matcher(P, T)
        for s in range(n - m + 1):
            if P[0:m] == T[s:s + m]:
                print('Pattern occurs with shift %d' % s)
    
    def rabin_karp_matcher(self, T : str, P : str, d, q):
        """
        Rabin-Karp字符串匹配算法
        """
        n = len(T)
        m = len(P)
        h = d ** (m - 1) % q
        p = 0
        t = 0
        for i in range(0, m):
            p = (d * p + ord(P[i]) - ord('0')) % q
            t = (d * t + ord(T[i]) - ord('0')) % q
        for s in range(0, n - m + 1):
            if p == t:
                if P[0:m] == T[s:s + m]:
                    print('Pattern occurs with shift %d' % s)
            if s < n - m:
                t = (d * (t - (ord(T[s]) - ord('0')) * h) + ord(T[s + m]) - ord('0')) % p
    
    def transition_function(self, q, Ti):
        """
        变迁函数d
        """
        return q

    def finite_automaton_matcher(self, T, d, m):
        """
        字符串匹配自动机的简易过程
        """
        n = len(T)
        q = 0
        for i in range(n):
            q = self.transition_function(q, T[i])
            if q == m:
                print('Pattern occurs with shift %d' % (i - m))

    def compute_transition_function(self, P, sigma):
        """
        下列过程根据一个给定模式`P[1..m]`来计算变迁函数`epsilon`, 运行时间为`O(m^3|∑|)`
        """
        m = len(P)
        for q in range(m + 1):
            for a in sigma:
                k = min(m + 1, q + 2)
                while P[k] != P[q]:
                    k -= 1
                epsilon = k
        return epsilon
    
    def compute_ptefix_function(self, P):
        """
        """
        m = len(P)
        pi = [0] * m
        k = 0
        for q in range(1, m):
            while k > 0 and P[k + 1] != P[q]:
                k = pi[k]
            if P[k + 1] == P[q]:
                k += 1
            pi[q] = k
        return pi

    def kmp_matcher(self, T, P):
        """
        Knuth-Morris-Pratt字符串匹配算法
        """
        n = len(T)
        m = len(P)
        pi = self.compute_ptefix_function(P)
        q = 0
        for i in range(n):
            while q >= 0 and P[q + 1] != T[i]:
                q = pi[q]
                if P[q + 1] == T[i]:
                    q = q + 1
                if q == m:
                    print('Pattern occurs with shift %d' (i - m))
                    q = pi[q]

    def repeat_factor(self, s):
        """
        求字符串中的重复因子
        """
        return list(map(lambda c : ord(c) ,s))

    def repetition_matcher(self, P, T):
        """
        """
        m = len(P)
        n = len(T)
        k = 1 + max(self.repeat_factor(P))
        q = 0
        s = 0
        while s <= n - m:
            if T[s + q + 1] == P[q + 1]:
                q += 1
                if q == m:
                    print('Pattern occurs with shift %d' % s)
            if q == m or T[s + q + 1] != P[q + 1]:
                s = s + max(1, q // k)
                q = 0

_inst = _StringMatch()

native_string_matcher = _inst.native_string_matcher
rabin_karp_matcher = _inst.rabin_karp_matcher
finite_automaton_matcher = _inst.finite_automaton_matcher
compute_transition_function = _inst.compute_transition_function
kmp_matcher = _inst.kmp_matcher

def test():
    """
    测试函数
    """
    native_string_matcher('eeabaaee', 'abaa')
    native_string_matcher('abc', 'dccabcd')
    native_string_matcher('3141592653589793', '26')
    rabin_karp_matcher('3141592653589793', '26', 10, 11)
    kmp_matcher('aabbcc', 'bb')

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter33/chapter33note.py
# python3 src/chapter33/chapter33note.py
"""

Class Chapter33_1

Class Chapter33_2

Class Chapter33_3

Class Chapter33_4

"""
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    pass
else:
    pass

class Chapter33_1:
    """
    chapter33.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter33.1 note

        Example
        ====
        ```python
        Chapter33_1().note()
        ```
        """
        print('chapter33.1 note as follow')
        print('第33章 计算几何学')
        print('计算几何学问题的输入一般是关于一组几何对象的描述,如一组点、一组线段,或者一个多边形的按逆时针顺序排列的一组顶点')
        print('输出常常是对有关这些对象的问题的回答,如是否直线相交,是否为一个新的几何对象,如顶点集合的凸包(convex hull,即最小封闭多边形)')
        print('本章中,将学习一些二维的(即平面上的)计算几何学算法,在这些算法中,每个输入对象都用一组点{p1,p2,p3,...}来表示,其中每个pi=(xi,yi)',
            'xi,yi∈R.例如：一个顶点的多边形P可以用一组点<p0,p1,p2,...,pn-1>来表示,这些点按照在P的边界上出现的顺序排列.')
        print('计算几何学也可以用来求解三维空间,甚至高维空间中的问题,但这样的问题及其解决方案是很难视觉化的')
        print('不过,即使是在二维平面上,也能够看到应用计算几何学技术的一些很好的例子')
        print('33.1节说明如何有效地准确回答有关线段的一些基本问题：一条线段是在与其共享一个端点的另一条线段的顺时针方向，还是在其逆时针方向')
        print('33.2节介绍一种称为“扫除”的技术，利用该技术设计一种运行时间为O(nlgn)的算法,用来确定n条线段中是否包含相交的线段')
        print('33.3节给出两种“旋转扫除”的算法，用于计算n个点的凸包(最小封闭的凸多边形)。这两个算法分别是运行时间为O(nlgn)的Graham扫描法和运行时间为O(nh)的Jarvis步进法(h是凸包中顶点的数目)')
        print('33.4节介绍一种运行时间为O(nlgn)的分治算法,用于在平面上的n个点中找出距离最近的一个点对')
        print('33.1 线段的性质')
        print('两个不同的点p1=(x1,y1)和p2=(x2,y2)的凸组合是满足下列条件的任意点p3=(x3,y3):对某个a,(0<=a<=1),有x3=ax1+(1-a)y2.',
            '也可以写作p3=ap1+(1-a)p2,从直观上看,p3是位于p1和p2的直线上、并处于p1和p2之间的凸组合的集合.我们称p1和p2为线段p1p2的端点。有时,还要考虑到p1和p2之间的顺序,这时,可以说有向线段p1p2',
            '如果p1是原点(0,0),则可以把有向线段p1p2看作向量p2')
        print('本节即将讨论以下问题')
        print('  (1)已知两条有向线段p0p1和p0p2,相对于它们的公共端点p0来说,p0p1是否在p0p2的顺时针方向上？')
        print('  (2)已知两条线段p0p1和p1p2,如果先通过p0p1再通过p1p2,在点p1处是不是要向左旋转')
        print('  (3)线段p1p2和p3p4是否相交')
        print('可以在O(1)时间内回答以上每个问题,这一点不会使人惊讶,因为每个问题的输入规模都是O(1),此外,将采用的方法仅限于加法、减法和比较运算',
            '既不需要除法运算，也不需要三角函数，这两者的计算代价都比较高昂,并且容易产生舍入误差等问题')
        print('例如要确定两条线段是否相交,一种直接的方法就是对这两条线段,都计算出形如y=mx+b的直线方程(其中m为斜率,b为y轴截距),找出两条直线的交点,',
            '并检查交点是否同时在两条线段上,在这一方法中,用除法求出交点.当线段接近于平行时,算法对实际计算机中除法运算的精度非常敏感',
            '本节中的方法避免使用除法,因而要精确的多')
        print('叉积')
        print('  叉积(cross product)的计算是关于线段算法的中心。向量p1和p2可以把叉积p1×p2看做是由点(0,0),p1,p2和p1+p2=(x1+x2,y1+y2)',
            '所形成的平行四边形的面积。另一种等价而更有用的定义是把叉积定义为一个矩阵的行列式',
            'p1×p2=det[[x1 x2],[y1 y2]]=x1y2-x2y1=-p2×p1')
        print('  如果p1×p2为正数,则相对于原点(0,0)来说,p1在p2在顺时针方向上;如果p1×p2为负数,则p1在p2的逆时针方向上')
        print('  为了确定相对于公共端点p0,有向线段p0p1是否在有向线段p0p2的顺时针方向,只需要把p0作为原点就可以了。亦即,可以用p1-p0表示向量p1’=(x1‘,y1’),其中x1‘=x1-x0,',
            'y1’=y1-y0.类似地可以定义p2-p0,然后计算叉积:',
            '(p1-p0)×(p2-p0)=(x1-x0)(y2-y0)-(x2-x0)(y1-y0)',
            '如果该叉积为正,则p0p1在p0p2的顺时针方向上;如果为负,则p0p1在p0p2的逆时针方向上')
        print('确定连续线段是向左转还是向右转')
        print('  在点p1处,两条连续的线段p0p1和p1p2是向左转还是向右转。亦即,希望找出一种方法,以确定一个给定的角∠p0p1p2的转向.运用叉积,',
            '使得无需对角进行计算,就可以回答这个问题.只需要检查一下有向线段p0p2是在有向线段p0p1的顺时针方向,还是在其逆时针方向.',
            '还是在其逆时针方向.在做这一判断时,要计算出叉积(p2-p0)×(p1-p0).如果该叉积的符号为负,则p0p2在p0p1的逆时针方向,因此,在p1点要向左转.',
            '如果叉积为正,就说明p0p2在p0p1的顺时针方向,因此,在点p1处要向右转。叉积为0说明点p0,p1和p2共线')
        print('确定两个线段是否相交')
        print('  为了确定两个线段是否相交,要检查每个线段是否跨越了包含另一线段的直线,给定一个线段p1p2,如果点p1位于某一直线的一边,而点p2位于该直线的另一边',
            '则称线段p1p2跨越了该直线.如果p1和p2就落在该直线上的话,即出现边界情况.两个线段相交,当且仅当下面两个条件中有一个成立,或同时成立')
        print('  (1) 每个线段都跨越包含了另一线段的直线')
        print('  (2) 一个线段的某一端点位于另一线段上.')
        print('叉积的其他应用')
        print('  33.3节中,根据相对于某一顶点原点的极角大小来对一组点进行排序')
        print('  33.2节中,运用红黑树来保持一组线段的垂直顺序。在这种方法中,',
            '并不是显式地记录关键字值,而是将红黑树代码中的每一次关键字值比较替换为叉积计算,以便确定与某指定垂直直线相交额两条线段中,相互的上下顺序')
        print('练习33.1-1 证明：如果p1*p2为正,则相对于原点(0,0),向量p1在向量p2的顺时针方向；如果叉积为负,则p1在p2的逆时针方向')
        print('练习33.1-2 略')
        print('练习33.1-3 一个点p1相对于原点p0的极角(polar angle)即在常规的极坐标系统中,向量p1-p0的极角.例如,相对于(2,4)而言,点(3,5)的极角即为向量(1,1)的极角',
            '即45度或Pi/4弧度.相对于(2,4)而言,(3,3)的极角为向量(1,-1)的极角,即315度或7Pi/4弧度.请编写一段伪代码,根据相对于某个给定原点p0的极角',
            '对一个由n点组成的序列<p1,p2,...,pn>进行排序.所给出过程的运行时间应为O(nlgn),并要求用叉积来比较极角的大小')
        print('练习33.1-4 试说明如何在O(n^2lgn)的时间内,确定n个点中的任意三点是否共线')                                     
        print('练习33.1-5 多边形(polygon)是平面上由一系列线段构成的、封闭的曲线。亦即，它是一条首尾相连的曲线，由一系列直线段所形成',
            '这些直线段称为多边形的边(side).一个连接了两条连续边的点称为多边形的顶点。如果多边形是简单的(一般情况下都会作此假设),',
            '它自身内部不会发生交叉。在平面上,由一个简单的多边形包围的一组点形成了该多边形的所有点则形成了其外部')
        print('练习33.1-6 已知一个点p0=(x0,y0),p0的右水平射线是点的集合{pi=(xi,yi):xi>=x0,且yi=y0},亦即,它是p0点正右方的点的集合,包含p0本身',
            '试说明如何通过把问题转换为确定两条线段是否相交的问题,从而可以在O(1)的时间内,确定给定的p0的右水平射线是否与线段p1p2相交')
        print('练习33.1-7 要确定一个点p0是否在一个简单多边形P(不一定是凸多边形)内部,一种方法是检查由p0发出的任何射线,看它是否与多边形P的边界相交奇数次',
            '但p0本身不能处于P的边界上.试说明如何在Θ(n)的时间内,计算出点p0是否在一个由n个顶点组成的多边形P的内部')
        print('练习33.1-8 试说明如何在Θ(n)的时间内,计算出由n个顶点所组成的简单多边形(但不一定是凸多边形)的面积')
        # python src/chapter33/chapter33note.py
        # python3 src/chapter33/chapter33note.py

class Chapter33_2:
    """
    chapter33.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter33.2 note

        Example
        ====
        ```python
        Chapter33_2().note()
        ```
        """
        print('chapter33.2 note as follow')
        print('33.2 确定任意一对线段是否相交')
        print('一组线段中任意两条线段是否相交的扫除技术,这种技术或其变体许多几何学算法都用到')
        print('该算法的运行时间为O(nlgn),其中n是已知线段的数目，它仅确定是否存在相交的线段,但并不输出所有的相交的线段')
        print('最坏情况下，要找出n个线段中的所有相交点，所需的时间为Ω(n^2)')
        print('在扫除过程中，一条假想的的垂直扫除线穿过已知的几何物体,并且通常是从左到右依次移动扫除线.',
            '扫除线移动的空间方向可以看作是一种时间上先后顺序的度量')
        print('扫除技术提供了一种对一组几何物体进行排序的方法,通常先把它们放入一个动态数据结构中,并且利用它们之间的关系对其进行排序.',
            '本节中确定线段相交的算法按从左到右的次序考察所有的线段端点,每遇到一个端点就检查是否是相交点')
        print('为了描述确定n条线段中任意两条是否相交的算法并证明其正确性,做出了如下两条简化性假设:第一,假定没有一条输入线段是垂直的.',
            '第二,假定没有三条输入线段相交于同一点.即使这两条假设不成立,算法也能正常的工作.的确,如果去掉上面的两条简化性假设后',
            '在为计算几何学算法编程并证明其正确性时,对边界条件的处理就常常是最棘手的部分了')
        print('排序线段')
        print('  假定不存在垂直线段,所以任何与给定垂直扫除线相交的输入线段与其只能有一个交点.因此,可以根据交点的y坐标对与给定垂直扫除线相交的线段进行排序')
        print('  更准确地说,考察两条线段s1和s2.如果一条横坐标为x的垂直扫除线与这两条线段都相交,则说两条线段在x是可比的',
            '如果s1和s2在x处是可比的,并且在x处,s1与扫除线的交点比s2与同一条扫除线的交点高,则说在x处s1位于s2之上,写作s1>xs2')
        # !扫除算法
        print('扫除线的移动')
        print('  典型的扫除算法要维护下列的两组数据')
        print('  (1) 扫除线状态：给出了与扫除线相交的物体之间的关系')
        print('  (2) 事件点调度：是一个从左向右排列的x坐标的序列,它定义了扫除线的暂停位置。',
            '称每个这样的暂停位置为事件点。扫除线状态仅在事件点处才会发生变化')
        print('对于某些算法,事件点调度是随算法执行而动态地确定的.现在讨论的算法仅是基于输入数据的简单性质静态地确定事件点。',
            '特别地,每条线段的端点都是事件点.通过增加x坐标,并从左向右执行来对线段的端点进行排序',
            '当遇到线段的左端点时,就把该线段插入到扫除线状态中,并且当遇到其右端点时,就把它从扫除线状态中删除.',
            '当每两条线段在全序中第一次变为连续时,就检查它们是否相交')
        print('扫除线状态是一个全序T,在T上要执行下列操作:')
        print('(1) INSERT(T, s): 把线段s插入到T中')
        print('(2) DELETE(T, s): 把线段s从T中删除')
        print('(3) ABOVE(T, s): 返回T中紧靠线段s上面的线段')
        print('(4) BELOW(T, s): 返回T中紧靠线段s下面的线段')
        print('如果输入中有n条线段,则可以运用红黑树,在O(lgn)时间内执行上述每个操作.读者可以回顾一下',
            '红黑树操作涉及了关键字的比较','此处可以用叉积比较来取代关键字比较,以确定两个线段的相对次序')
        print('求线段交点的伪代码')
        print('ANY-SEGMENTS-INTERSECT(S)下列算法的输入是由n个线段组成的集合S,如果S中的任何一对线段相交,',
            '算法就返回布尔值TRUE,否则就返回就FALSE.全序T是由是由一棵红黑树来实现的')
        print('正确性')
        print('  证明当且仅当S中的线段有一个交点时,对ANY-SEGMENTS-INTERSECT(S)的调用返回TRUE')
        print('运行时间')
        print('  如果集合S中有n条线段,则ANY-SEGMENTS-INTERSECT的运行时间为O(nlgn),',
            '则可以使每次相交测试所需的时间为O(1).因此,总的运行时间为O(nlgn)')
        print('练习33.2-1 证明:在n条线段的集合中,可能有Θ(n^2)个交点')
        print('练习33.2-2 已知两条在x处可比的线段a和b,式说明如何在O(1)时间内确定a>xb和b>xa中哪一个成立.假定这两条线段都不是垂直的',
            '(提示：如果a和b不相交,仍然可以只利用加、减、乘这几种运算，无需用除法.当然,在应用>x关系时,如果a和b相交,就可以停下来并声明找到了一个交点)')
        print('练习33.2-3 Maginot教授建议修改过程ANY-SEGMENTS-INTERSECT,使其不是找出一个交点后就返回,而是输出相交的线段,',
            '再继续进行for循环的下一次迭代.称这样得到的过程为PRINT-INTERSECTING-SEGMENTS,并声称该过程能够安札线段在集合中出现的次序',
            '从左到右输出所有的交点.试着说明这位教授的说法有两点是错误的,即举出一组线段,使得运用过程PRINT-INTERSECTING-SEGMENTS所找出的一个相交点不是最左相交点',
            '再举出出一组线段,使过程PRINT-INTERSECGING-SEGMENTS不能找出所有的相交点')
        print('练习33.2-4 写出一个运行时间为O(nlgn)的算法,以确定由n个顶点组成的多边形是否是简单多边形')
        print('练习33.2-5 写出一个运行时间为O(nlgn)的算法,以确定总共有n个顶点的两个简单多边形是否相交')
        print('练习33.2-6 一个圆面是由一个圆加上其内部所组成,并且用圆心和半径表示。如果两个圆面有任何公共点',
            '则称这两个圆面相交。写出一个运行时间为O(nlgn)的算法,一确定n个圆面中是否有任何两个圆面相交')
        print('练习33.2-7 已知n条线段中总共包含k个交点,式说明如何在O((n+k)lgn)时间内,输出全部k个交点')
        print('练习33.2-8 论证即使有三条或更多的线段相交于同一点,过程ANY-SEGMENTS-INTERSECT也能正确执行')
        print('练习33.2-9 证明在有垂直线段的情况下,如果某一垂直线段的底部端点被当做是左端点来处理,其顶部端点被当做是右端点来处理,',
            '则过程ANY-SEGMENTS-INTERSECT也能正确执行.如果允许有垂直线段的话')
        # python src/chapter33/chapter33note.py
        # python3 src/chapter33/chapter33note.py

class Chapter33_3:
    """
    chapter33.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter33.3 note

        Example
        ====
        ```python
        Chapter33_3().note()
        ```
        """
        print('chapter33.3 note as follow')
        print('33.3 寻找凸包')
        print('点集Q的凸包是一个最小的凸多边形P,满足Q的每个点或者在P的边界上，或者在P的内部')
        print('用CH(Q)表示Q的凸包')
        print('两种算法用于计算包含n个点的点集的凸包,第一种算法称为Graham扫描法,运行时间为O(nlgn),',
            '第二种是Jarvis步进法,运行时间为O(nh),h为凸包中的顶点数')
        print('CH(Q)的每一个顶点都是Q中的点,两种算法都用了这种性质,来决定应该保留Q中的哪些点作为凸包的顶点,以及应该去掉Q中的哪些点')
        print('事实上,有好几种方法都能在O(nlgn)时间内计算凸包,两种算法都运用了旋转扫描的技术',
            '根据每一个顶点对一个参照顶点的极角的大小,依次进行处理,其他方法有一下几种')
        print('  (1) 增量方法:对点从左到右进行排序后,得到一个序列<p1,p2,...,pn>,在第i步,根据左起的第i个点',
            '对i-1个最左边的点的凸包CH({p1,p2,...,pi-1})进行更新,从而形成凸包CH({p1,p2,...,pi})',
            '实现这种方法所需的全部时间为O(nlgn)')
        print('  (2) 分治法:在Θ(n)时间内,n个点组成的集合被划分为两个子集,分别包含最左边的[n/2]个点和最右边的[n/2]个点',
            '并对子集的凸包进行递归计算,然后利用一种巧妙的方法,在O(n)时间内对计算出来的凸包进行组合',
            '这种方法的计算时间由递归式T(n)=2T(n/2)+O(n)表示,因此分治法的运行时间为O(nlgn)')
        print('  (3) 剪枝-搜索方法:类似于最坏情况下线性时间的中值算法,通过反复丢弃剩余点中固定数量的点,来寻找凸包的上部(上链),',
            '直至只剩下凸包的上部,再执行同样的操作找到凸包的下链,从渐进意义上看,这种方法的速度最快,如果包含凸包包含h个顶点的话',
            '该方法的运行时间仅为O(nlgh)')
        print('计算一组点的凸包本身就是一个有趣的问题.其他一些关于计算几何学问题的算法都始于对凸包的计算.',
            '例如,考虑二维的最远点对问题：已知平面上的n个点的集合,希望找出它们中彼此之间距离最远的两个点')
        print('要找出n个顶点的凸多边形中最远顶点对,需要O(n)的时间.因此,通过在O(nlgn)时间内计算出n个输入点的凸包,然后再找出得到的凸多边形中的最远顶点对,',
            '就可以在O(nlgn)的时间内,找出任意n个点组成的集合中距离最远的点对')
        print('Graham扫描法')
        print('  Graham扫描法通过设置一个关于候选点的堆栈S来解决凸包问题.输入集合Q中的每个点都被压入栈一次,非CH(Q)中顶点的点最终将被弹出堆栈',
            '当算法终止时,堆栈S中仅包含CH(Q)中的顶点,其顺序为各点在边界上出现的逆时针方向排列的顺序')
        print('  过程GRAHAM-SCAN的输入为点集Q,|Q|>=3,它调用函数TOP(S),以便在不改变堆栈S的情况下,',
            '返回处于栈顶的点,并调用函数NEXT-TO-TOP(S)来返回处于堆栈顶部下面的那个点,且不改变栈S',
            '可以证明:过程GRAHAM-SCAN返回的堆栈S从底部到顶部,依次是按逆时针方向排列的CH(Q)中的顶点')
        print('  过程GRAHAM-SCAN的执行过程:首先选取p0作为y坐标最小的点,如果有数个这样的点,则选取最左边的点作为p0.由于Q中没有其他点比p0更低,',
            '并且与其有相同y坐标的点都在它的右边,所以p0是CH(Q)的一个顶点.再根据Q中剩余的点相对于p0的极角对它们进行排序,使用比较叉积',
            '如果有两个或者更多的点相对于p0的极角相同,那么除了与p0距离最远的点以外,其余各点都是p0与该最远点的凸组合',
            '因此,可以完全不考虑这些点。设m表示除p0以外剩余的点的数目')
        print('定理33.1 (Graham扫描法的正确性) 如果在一个点集Q上运行GRAHAM-SCAN,其中|Q|>=3,则在过程终止时,栈S从底到顶,',
            '按逆时针方向顺序包含了CH(Q)中的各个顶点')
        print('循环不变式：初始化，保持，终止')
        print('Jarvis步进法')
        print('  Jarvis步进法运用了一种称为打包的技术来计算一个点集Q的凸包.算法的运行时间为O(nh),其中h是CH(Q)中的顶点数',
            '当h为o(lgn)时,Jarvis步进法在渐进意义上比Graham扫描法的速度更快些')
        print('  从直观上看,可以把Jarvis步进法想象成在集合Q的外面紧紧包了一层纸。开始时把纸的末端粘在集合中最低的点上,',
            '即粘在与Graham扫描法开始时相同的点p0上,该点为凸包的一个顶点.把纸拉向右边使其紧绷,然后再把纸拉高一些,直到碰到一个点.',
            '该点必定是凸包中的一个顶点.使纸保持紧绷状态,用这种方法继续围绕顶点集合,直至回到原始点p0')
        print('  如果有适当的实现方法,Jarvis步进法的运行时间就会是O(nh),对CH(Q)的h个顶点中的每一个顶点,都找出具有极小极角的顶点',
            '如果每次极角比较操作所需的时间为O(1),则可以在O(n)时间内计算出n个值中的最小值.因此,Jarvis步进法的运行时间为O(nh)')
        print('练习33.3-1 证明：在过程GRAHAM-SCAN中,点p1和pm必定是CH(Q)的顶点')
        print('练习33.3-2 考虑一个能支持加法、比较和乘法的计算模型，用该模型对n个数进行排序时,存在一个下界Ω(nlgn).证明：',
            '当在这样一个模型中有序的计算出n个点组成的集合的凸包时,其下界为Ω(nlgn)')
        print('练习33.3-3 已知一组点的集合Q,证明彼此间距离最远的点对必定是CH(Q)中的顶点')
        print('练习33.3-4 对给定的一个多边形P和在其边界上的一个点q,q的阴影是满足线段qr完全在P的边界上或内部的点r的集合。',
            '如果在P的内部存在一个点p的集合称为P的内核。给定一个n个顶点的星形多边形P按逆时针方向排序的各个顶点,试说明如何在O(n)的时间内计算出CH(Q)')
        print('练习33.3-5 在联机凸包问题中,每次只给出n个点组成的集合Q中的一个点.在接收到每个点后,就计算出目前所见到的点的凸包.显然,可以对每个点运行一次Graham扫描算法',
            '总的运行时间为O(n^2lgn).试说明如何在O(n^2)时间内解决联机凸包问题')
        print('练习33.3-6 试说明如何实现增量方法,使其在O(nlgn)的时间内,计算出n个点的凸包')
        # python src/chapter33/chapter33note.py
        # python3 src/chapter33/chapter33note.py

class Chapter33_4:
    """
    chapter33.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter33.4 note

        Example
        ====
        ```python
        Chapter33_4().note()
        ```
        """
        print('chapter33.4 note as follow')
        print('33.4 寻找最近点对')
        print('考虑一下在n>=2个点的集合Q中寻找最近点对的问题。集合Q中两个点可能重合,在这种情况下,它们之间的距离为0.这一问题可以应用于交通控制等系统中',
            '在空中或者海洋交通控制系统中,需要发现两个距离最近的交通工具,以便检测出可能发生的相撞事故')
        print('在最简单的蛮力搜索最近点对的算法中,要查看所有Θ(n^2)个点对.可以采用分治算法解决该问题,',
            '其运行时间为递归式T(n)=2T(n/2)+O(n)来描述,算法运行时间仅为O(nlgn)')
        print('分治算法')
        print('  算法的每一次递归调用的输入为子集P∈Q和数组X和Y,每个数组均包含输入子集P的所有点',
            '对数组X中的点,按其x坐标单调递增的顺序进行排序。类似地,对数组Y中的点按其y坐标单调递增的顺序进行排序.',
            '注意，为了获得O(nlgn)的时间界,不能在每次递归调用中都进行排序.如果每次递归调用都进行排序的话,',
            '运行时间的递归式就变为T(n)=2T(n/2)+O(nlgn),其解为T(n)=O(nlgnlgn)')
        print('  对X坐标排序(分治准备)')
        print('  找到左半边中的最近点对dL (解决子问题)')
        print('  找到右半边中的最近点对dR (解决子问题)')
        print('  找到跨左右两边的最近点对dC (合并)')
        print('合并：找到跨左右两边的最近点对dC')
        print('  设d=min(dL,dR)')
        print('只需检查距离中线范围d范围内的点')
        print('但还是不能蛮力检查该范围内的所有点：仍然太多')
        print('观察：对于每个点来说,只需看y坐标也在d以内的点.这样的话最多跟7个点比较')
        print('练习33.4-1 Smothers教授提出了一个方案,即在最近点对算法中,只检查数组Y\'中每个点后面的5个点,其思想是总是把直线l上的点放入集合PL',
            '那么,直线l上就不可能有一个点属于PL,另一个点属于PR的重合点对.因此,至多可能有6个点处于d*2d的矩形内.这种方案的缺陷何在')
        print('练习33.4-2 在不增加算法渐进运行时间的前提下,试说明如何保证传递给第一次递归调用的点集中不包含重合的点',
            '证明这样一来,只需要检查数组Y\'中跟随每个点后的5个数组位置就足够了?')
        print('练习33.4-3 两个点之间的距离除欧几里得距离外,还有其他定义方法.在平面上,点p1和p2之间的Lm距离由下式给出.',
            '因此,欧几里得距离实际上是L2距离.修改最近点对算法,使其利用L1距离,也称为曼哈顿距离')
        print('练习33.4-4 已知平面上的两个点p1和p2,它们之间的距离L∞距离为max(|x1-x2|,|y1-y2|),修改最近点对算法,使其能利用L∞距离')
        print('练习33.4-5 对最近点对算法进行修改,使其能避免对数组Y进行预排序,但仍然能使算法的运行时间保持为O(nlgn)')
        print('思考题33-1 凸层')
        print('  已知平面上的点集Q,用归纳法来定义Q的凸层(convex layer).Q的第一凸层是由Q中是CH(Q)顶点的那些点组成.对i>1,定义Qi由把Q中所有在凸层1,2,...,i-1',
            '中的点去除后剩余的点所组成.如果Qi!=p,那么Q的第i凸层为CH(Qi);否则,第i凸层无定义')
        print('  a) 写出一个运行时间为O(n^2)的算法,以找出n个点所组成的集合的各凸层')
        print('  b) 证明:在对n个实数进行排序所需时间为Ω(nlgn)的任何计算模型上,要计算出n个点凸层需要Ω(nlgn)时间')
        print('思考题33-2 最大层')
        print('  设Q是平面上n个点所组成的集合.如果有x>=x\'且y>=y\',则称点(x,y)支配点(x\',y\').Q中不被其中任何其他点支配的点称为最大点',
            '注意,Q可以包含许多最大点,可以把这些最大点组织成如下的最大层')
        print('  描述一种时间为O(nlgn)的算法,以便计算出n个点的集合Q的各最大层(提示:把一条扫除线从右向左移动)')
        print('  如果允许输入点有相同的x坐标或y坐标,会不会出现问题')
        print('思考题33-3 魑魅和鬼问题')
        print('  有n个巨人与n个鬼战斗.每个巨人的武器是一个质子包,它可以用一串质子流射中鬼而把鬼消灭。质子流沿直线行进,在击中鬼时就终止',
            '巨人决定采取下列策略.他们各自寻找一个鬼形成n个巨人-鬼对,然后每个巨人同时向各自选取的鬼射出一串质子流.',
            '并且巨人选择的配对方式应该使质子流都不会交叉')
        print('  假定每个巨人和每个鬼的位置都是平面上一个固定的点,并且没有三个位置共线')
        print('  a) 论证存在一条通过一个巨人和一个鬼的直线,使得直线一边的巨人数与同一边的鬼数相等.试说明如何在O(nlgn)时间内找出这样一条直线')
        print('  b) 写出一个运行时间为O(n^2lgn)的算法,使其按不会有质子流交叉的条件把巨人与鬼配对')
        print('思考题33-4 拾取棍子问题')
        print('  有n根小棍子,以某种方式,互相叠放在一起.每根棍子都用其端点来指定,每个端点都是一个有序的三元组,给出了其(x,y,z)坐标.',
            '所有棍子都不是垂直的.希望拾取所有的棍子,但要满足条件,一次一根当一根棍子上面没有压着其他根子时,才可以挑起该棍子')
        print('  a) 给出一个过程,取两根棍子a和b作为参数,报告a是在b的上面、下面还是与b无关')
        print('  b) 给出一个有效的算法,它应能确定是否有可能拾取所有的棍子.如果能,提供一个拾取所有棍子的合法顺序')
        print('思考题33-5 稀疏包分布问题')
        print('  考虑计算平面上点的集合的凸包问题,但这些点是根据某已知的随机分布取得的.有时,从这样一种分布取得的n个点的凸包的期望规模为O(n^(1-e))',
            '其中e为大于0的某个常数.称这样的分布为稀疏包分布。稀疏包分布包括以下几种:')
        print('  点是均匀地从一个单位半径的圆面中取得的,凸包的期望规模为Θ(n^1/3)')
        print('  点是均匀地从一个具有k条边的凸多边形内部取得的(k为任意常数).凸包的期望规模为Θ(lgn)')
        print('  点是根据二维正态分布取得的.凸包的期望规模为Θ(sqrt(lgn))')
        print('  a) 已知两个分别有n1和n2个顶点的凸多边形,说明如何在O(n1+n2)时间内,计算出全部n1+n2个点的凸包(多边形可以重叠)')
        print('  b) 证明:对于根据稀疏包分布独立取得的一组n个点,其凸包可以在O(n)的期望时间内计算出来.',
            '(提示：采用递归方法分别求得出前n/2个点和后n/2个点的凸包,然后再对结果进行合并)')
        # python src/chapter33/chapter33note.py
        # python3 src/chapter33/chapter33note.py

chapter33_1 = Chapter33_1()
chapter33_2 = Chapter33_2()
chapter33_3 = Chapter33_3()
chapter33_4 = Chapter33_4()

def printchapter33note():
    """
    print chapter33 note.
    """
    print('Run main : single chapter thirty-three!')
    chapter33_1.note()
    chapter33_2.note()
    chapter33_3.note()
    chapter33_4.note()

# python src/chapter33/chapter33note.py
# python3 src/chapter33/chapter33note.py

if __name__ == '__main__':  
    printchapter33note()
else:
    pass

```

```py

class Point:
    """
    点 `(x, y)`
    """
    def __init__(self, x = 0, y = 0):
        """
        点 `(x, y)`
        """
        self.x = x
        self.y = y
    
    def location(self):
        """
        Return
        ===
        (`x`, `y`)
        """
        return self.x, self.y

class _ComputedGeometry:
    """
    计算几何学算法
    """
    def __init__(self):
        """
        计算几何学算法
        """
        pass

    def direction(self, pi : Point, pj : Point, pk : Point):
        """
        Return
        ===
        `(pj - pi) × (pk - pi)`
        """
        xi, yi = pi.location()
        xj, yj = pj.location()
        xk, yk = pk.location()
        return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)

    def on_segment(self, pi : Point, pj : Point, pk : Point):
        """
        点`pi`是否在线段`pjpk`上
        """
        xi, yi = pi.location()
        xj, yj = pj.location()
        xk, yk = pk.location()
        if (min(xi, xj) <= xk and xk <= max(xi, xj)) and (min(yi, yj) <= yk and yk <= max(yi, yj)):
            return True
        return False

    def segments_intersect(self, p1 : Point, p2 : Point, p3 : Point, p4 : Point):
        """
        判断线段p1p2和p3p4是否相交，相交返回Ture否则返回False
        """
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
        if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0) and \
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0):
            return True
        elif d1 == 0 and self.on_segment(p3, p4, p1) == True:
            return True
        elif d2 == 0 and self.on_segment(p3, p4, p2) == True:
            return True
        elif d3 == 0 and self.on_segment(p1, p2, p3) == True:
            return True
        elif d4 == 0 and self.on_segment(p1, p2, p4) == True:
            return True
        return False

_inst = _ComputedGeometry()
segments_intersect = _inst.segments_intersect

def test():
    """
    测试函数
    """
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(1, 0)
    p4 = Point(0, 1)
    print('线段p1p2和线段p3p4是否相交', segments_intersect(p1, p2, p3, p4))
    print('线段p1p3和线段p2p4是否相交', segments_intersect(p1, p3, p2, p4))

if __name__ == '__main__':
    test()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter34/chapter34note.py
# python3 src/chapter34/chapter34note.py
"""

Class Chapter34_1

Class Chapter34_2

Class Chapter34_3

Class Chapter34_4

Class Chapter34_5

"""
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    pass
else:
    pass

class Chapter34_1:
    """
    chapter34.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.1 note

        Example
        ====
        ```python
        Chapter34_1().note()
        ```
        """
        print('chapter34.1 note as follow')
        print('第34章 NP完全性')
        print('之前学习过的算法都是多项式时间算法:对规模为n的输入,它们在最坏情况下的运行时间为O(n^k),其中k为某个常数',
            '但是不是所有的问题都能在多项式时间内解决.例如图灵著名的“停机问题”,任何计算机不论耗费多少时间也不能解决')
        print('还有一些问题是可以解决的,但对任意常数k,都不能在O(n^k)的时间内得到解答.')
        print('一般来说,把可以由多项式时间的算法解决的问题看作是易处理的问题,而把需要超多项式时间才能解决的问题看作是难处理的问题')
        print('本章的主题是一类称为“NP完全”(NP-complete)的有趣问题,它的状态是未知的.直到现在,既没有人找出求解NP完全问题的多项式算法,',
            '也没有人能够证明对这类问题不存在多项式时间算法')
        print('这一所谓的P!=NP问题自1971年被提出之后,已经成为了理论计算机科学研究领域中,最深奥和最错综复杂的开放问题之一了')
        print('NP完全问题有一个方面特别诱人,就是在这一类问题中,有几个问题在表面上看来与有着多项式时间算法的问题非常相似',
            '在下面列出的每一对问题中,一个是可以在多项式时间内求解的,另一个是NP完全的,但从表面上看,两个问题之间差别很小')
        print('最短与最长简单路径：在第24章中,即使是在有负权值边的情况下,也能在一个有向图G=(V,E)中,在O(VE)的时间内,从一个源顶点开始找出最短的路径',
            '然而,寻找两个顶点间最长简单路径是NP完全的.事实上,即使所有边的权值都是1,它也是NP完全的')
        print('欧拉游程与哈密顿回路：对一个连通的有向图G=(V,E)的欧拉游程是一个回路,它遍历图G中每条边一次',
            '但可能不止一次地访问同一个顶点,可以在O(E)时间内,确定一个图中是否存在一个欧拉游程,并且能在O(E)时间内找出这一欧拉游程中的各条边',
            '一个有向图G=(V,E)的哈密顿回路是一种简单回路,它包含V中的每个顶点。确定一个有向图或者无向图中是否存在哈密顿回路是NP完全的')
        print('NP完全性与P类和NP类')
        print('  三类问题：P,NP和NPC,其中最后一类指NP完全问题.此处只是非形式地对它们进行描述,后面将给出更为形式的定义')
        print('  P类中包含的是在多项式时间内可解的问题.更具体一点,都是一些可以在O(n^k)时间内求解的问题,此处k为某个常数,n是问题的输入规模',
            '前面大多数算法绝大部分都属于P类')
        print('  NP类中包含的是在多项式时间内“可验证”的问题,此处是指某一解决方案的“证书”,就能够在问题输入规模的多项式时间内,验证该证书是正确的',
            '例如:在哈密顿回路问题中,给定一个有向图G=(V,E),证书可以是一个包含|V|个顶点的序列<v1,v2,v3,...,v|v|>',
            '很容易在多项式时间内判断出对i=1,2,3,...,|V|-1,是否有(vi,vi+1)∈E和(v|v|,v1)∈E')
        print('  P中的任何问题也都属于NP,这是因为如果某一问题是属于P的,则可以在不给出证书的情况下,在多项式时间内解决它.',
            '至于P是否是NP的一个真子集,在目前是一个开放的问题')
        print('  从非形式的意义上来说,如果一个问题属于NP,且与NP中的任何一个问题是一样“难的”(hard),写说它属于NPC类,也称它为NP完全的')
        print('  同时,不加证明地宣称如果任何NP完全的问题可以在多项式时间内解决,则每一个NP完全的问题都有一个多项式时间的算法')
        print('  从表面上看,很多自然而有趣的问题并不比排序,图的搜索或网络流问题更困难,但事实上,它们却是NP完全问题')
        print('NP完全性证明有点类似于8.1节中任何比较排序算法的运行时间下界Ω(nlgn)的证明;',
            '但是用于证明NP完全性所用到的特殊技巧却和8.1节中的决策树方法是不同的')
        print('证明一个问题为NP完全问题时,要依赖于三个关键概念：')
        print('  (1) 判定问题与最优化问题')
        print('  很多有趣的问题都是最优化问题(optimization problem),其中每一种可能的解都有一个相关的值,我们的目标是找出一个具有最佳值的可行解',
            '例如,在一个称为SHORTEST-PATH,已知的是一个无向图G及顶点u和v,要找出u到v之间的经过最少边的路径',
            '(换句话说,SHORTEST-PATH是在一个无权、无向图中的单点对间最短路径问题),',
            '然而,NP完全性不直接适合于最优化问题,但适合于判定问题(decision problem)',
            '因为这种问题的答案是简单的“是(1)”或“否(0)”')
        print('  通常,通过对待优化的值强加一个界,就可以将一个给定的最优化问题转换为一个相关的判定问题了')
        print('  例如,对SHORTEST-PATH问题来说,它有一个相关性的判定问题(称其为PATH),就是要判定给定的有向图G,顶点u和v,一个整数k,在u和v之间是否存在一条包含至多k条边的路径')
        print('  试图证明最优化问题是一个“困难的”问题时,就可以利用该问题与相关的判定问题之间的关系.',
            '这是因为,从某种意义上来说,判定问题要“更容易一些”,或至少“不会更难”')
        print('  例如,可以先解决SHORTEST-PATH问题,再将找出的最短路径上边的数目与相关判定问题中参数k进行比较,从而可以解决PATH问题',
            '换句话说,如果某个最优化问题比较容易的话,那么其相关的判定问题也会是比较容易的')
        print('  (2) 归约')
        print('  上述有关证明一个问题不难于或不简单于另一个问题的说法,对两个问题都是判定问题也是适用的',
            '在几乎每一个NP完全性证明中,都利用了这一思想.做法如下:考虑一个判定问题(称为A),希望在多项式时间内解决该问题.',
            '称某一特定问题的输入为该问题的一个实例,例如,PATH问题的一个实例可以是某一特定的图G、G中特定的点u和v和一个特定的整数k',
            '现在,假设有另一个不同的判定问题(称为问题B),知道如何在多项式时间内解决它.假设有一个过程,能将A的任何实例a转换成B的、具有以下特征的某个实例b:')
        print('  1) 转换操作需要多项式时间;')
        print('  2) 两个实例的答案是相同的.亦即,a的答案是“是”,当且仅当b的答案也是“是”')
        print('  称这样的一种过程为多项式时间的归约算法(reduction algorithm),并且提供了一种在多项式时间内解决问题A的方法:')
        print('  1) 给定问题A的一个实例a,利用多项式归约算法,将它转换为问题B的一个实例b')
        print('  2) 在实例b上,运行B的多项式时间判定算法')
        print('  3) 将b的答案用作a的答案')
        print('  只要上述步骤中的每一步只需多项式时间,则所有三步合起来也只需要多项式时间,这样就有了一种对a进行判断的方法')
        print('  换句话说,通过将对问题A的求解“归约”为对问题B的求解,就可以利用B的“易求解性”来证明A的“易求解性”')
        print('NP完全性是为了反映一个问题有多难,而不是为了反映它是多么容易.',
            '因此,以相反的方式来利用多项式时间归约,从而说明某一问题是NP完全的')
        print('至于NP完全性,不能假设问题A绝对没有多项式时间的算法,然而,证明的方法是类似的,即假设问题A是NP完全的前提下,来证明问题B是NP完全的')
        print('第一个NP完全问题')
        print('  应用归约技术要有一个前提,即已知一个NP完全的问题,这样才能证明另一个问题也是NP完全的.',
            '因此,需要找到一个“第一个NP完全问题”.将使用的这一个问题就是电路可满足性问题')
        print('  在这个问题中,已知的是一个布尔组合电路,它由AND、OR和NOT门组成,希望知道这个电路是否存在一组布尔输入,能够使它的输出为1')
        print('34.1 多项式时间')
        print('多项式时间可解问题的形式化定义')
        print('  (1) 把所需运行时间Θ(n^100)的问题作为难处理问题的合理之处,但在实际中,需要如此高次的多项式时间的问题是非常少的',
            '在实际中,所遇到的典型多项式时间可解问题所需的时间要少的多.经验表明：一旦某一问题的一个多项式时间算法被发现后,',
            '往往就会发现一些更为有效的算法,即使对某个问题来说,当前最佳算法的运行时间为Θ(n^100),很有可能在很短的时间内',
            '就能找到一个运行时间要好的多的算法')
        # !串行随机存取计算机模型
        print('  (2) 对很多合理的计算模型来说,在一个模型上用多项式时间可解的问题,在另一个模型上也可以在多项式时间内获得解决',
            '例如,在串行随机存取计算机模型上多项式可求解的问题类,与抽象的图灵机上在多项式时间内可求解的问题类是相同的,',
            '它也与利用并行计算机在多项式时间内可求解的问题类相同,即使处理器数目随输入规模以多项式增加也是这样')
        print('  (3) 多项式时间可解问题问题类具有很好的封闭性,这是因为在加法、乘法和组合运算下,多项式是封闭的.',
            '例如,如果一个多项式时间算法的输出馈送给另一个多项式时间算法作为输入,则得到的组合算法也是多项式时间的算法',
            '如果另外一个多项式时间算法对一个多项式时间的子程序进行常数次调用,那么组合算法的运行时间也是多项式的')
        print('抽象问题')
        print('  抽象问题Q为在问题实例集合I和问题解法集合S上的一个二元关系.例如,SHORTEST-PATH的一个实例是由一个图和两个顶点所组成的三元组,',
            '其解为图中的顶点序列,序列可能为空,表示两个顶点间不存在通路.问题SHORTEST-PATH本身就是一个关系,',
            '把图的每个实例和两个顶点与图中联系这两个顶点的最短路径联系在了一起.因为最短路径不一定是唯一的.',
            '因此,一个给定问题实例可能有多个解')
        print('编码')
        print('  如果要用一个计算机程序求解一个抽象问题,就必须用一种程序能理解的方式来表示问题实例.')
        print('  抽象对象集合S的编码是从S到二进制串集合的映射e.例如,都熟悉把自然数N={0,1,2,3,4,...}编码为{0,1,10,11,100,101,...}',
            '如bin(17)=10001')
        print(' ASCII编码、EBCDIC编码、UNICODE编码')
        print('多边形、图、函数、有序对、程序等所有这些都可以编码为二进制串')
        print('因此,“求解”某个抽象判定问题的计算机算法实际上把一个问题实例我的编码作为其输入.',
            '把实例集为二进制串的集合的问题称为具体问题.当提供给一个算法的是长度n=|i|的一个问题实例i时,算法可以在O(T(n))时间内产生问题的解,',
            '就说该算法在时间O(T(n))内解决了该具体问题。')
        print('因此，如果对某个常数k，存在一个算法能在时间O(n^k)内求解出某具体问题,就说该具体问题是多项式时间可解的')
        print('根据编码的不同,算法的运行时间可以是多项式时间或超多项式时间')
        print('如果不先指定编码,就不可能真正谈及对一个抽象的问题求解.在实际中,如果不采用“代价高昂的”编码(一元编码)',
            '则问题的实际编码形式对问题是否能在多项式时间内求解是微不足道的',
            '例如,以3代替2为基数来表示整数,对问题是否能在多项式时间内求解没有任何影响,',
            '因为可以在多项式时间内,将以3表示的整数可以转换为以基数2表示的整数')
        print('对一个抽象问题的实例,采用二进制或三进制来进行编码,对其“复杂性”都没有影响')
        print('引理34.1 设Q是定义在一个实例集I上的一个抽象判定问题,e1和e2是I上多项式相关的编码,则e1(Q)∈P当且仅当e2(Q)∈P')
        print('只要隐式地使用与标准编码多项式相关的编码,就可以直接讨论抽象问题,而无需参照任何特定的编码',
            '因为已经知道选取哪一种编码对该问题是否多项式时间可解没有任何影响')
        print('形式语言体系')
        print('  判定问题PATH对应的语言为')
        print('  PATH={<G,u,v,k>：G=(V,E)是一个无向图,u,v∈V,k>=0是一个整数,图G中从u到v存在一条长度至多为k的路径}')
        print('  形式语言体系可以用来表述判定问题与求解这些问题的算法之间的关系.如果对给定输入x,算法输出A(x)=1,就说算法A接受串x∈{0,1}.',
            '被算法A接受的语言是串的集合L={x∈{0,1}*:A(x)=1},即为算法所接受的串的集合。如果A(x)=0,则说算法A拒绝串x')
        print('  例如,语言PATH就能够多项式时间内被接受.一个多项式时间的接受算法要验证G是否编码一个无向图G,u和v是否是G中的顶点,',
            '利用广度优先搜索计算出G中从u到v的最短路径,然后把得到的最短路径上边数与k进行比较',
            '如果G编码了无向图,且从u到v的路径中至多有k条边,则算法输出1并停机;否则,该算法永远运行下去',
            '但是这一算法并没有对PATH问题进行判定,因为对最短路径长度多余k条边的实例,算法并没有显示地输出0',
            'PATH的判定算法必须显式地拒绝不属于PATH的二进制串.',
            '对PATH这样的判定问题来说,很容易设计出这样的一种判定算法：当不存在从u到v的、包含至多k条边的路径时,算法不是永远的运行下去,',
            '而是输出0并停机.对于其他的一些问题(如图灵停机问题),只存在接受算法,而不存在判定算法')
        print('可以非形式地定义一个复杂性类(complexity class)为语言的一个集合,某一语言是否属于该集合,可以通过某种复杂性度量(complexity measure)来确定',
            '比如一个算法的运行时间,该算法可以确定某个给定的串x是否属于语言L.',
            '复杂性类的实际定义要更专业一些')
        print('运用上述的形式语言理论体系,可以提出关于复杂性类P的另外一种定义:')
        print(' P∈{L∈{0,1}*：存在一个算法A能在多项式时间内判定L}.事实上,P也是能在多项式时间内被接受的语言类')
        print('定理34.2 P={L：L能被一个多项式时间的算法所接受}')
        print('练习34.1-1 定义最优化问题LONGEST-PATH-LENGTH为一个关系,它将一个无向图的每个实例、两个顶点与这两个顶点间最长路径中所包含的边数联系了起来',
            '定义判定问题LONGEST-PATH={<G,u,v,k>:G=(V,E)为一个无向图,u,v∈V,k>=0是一个整数,且G中存在着一条从u到v的简单路径,它包含了至少k条边}.',
            '证明最优问题LONGEST-PATH-LENGTH可以在多项式时间内解决,当且仅当LONGEST-PATH∈P')
        print('练习34.1-2 对于在无向图中寻找最长简单回路这一问题,给出其形式化的定义及相关的判定问题.另外,给出与该判定问题对应的语言')
        print('练习34.1-3 给出一种形式化的编码,利用邻接矩阵表示形式,将有向图编码为二进制串.另外再给出利用邻接表表示的编码.论证这两种表示是多项式时间相关的')
        print('练习34.1-4 0-1背包问题的动态规划算法是一个多项式时间算法')
        print('练习34.1-5 证明:对于一个多项式时间算法,当它调用一个多项式时间的子例程时,如果至多调用常数次,则此算法以多项式时间运行',
            '但是,当调用子例程的次数为多项式时,此算法就可能变成一个指数时间的算法')
        print('练习34.1-6 类P在被看作是一个语言集合时,在并集、交集、补集和Kleene星运算下是封闭的.',
            '亦即,如果L1,L2∈P,则L1∪L2∈P,等等.')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_2:
    """
    chapter34.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.2 note

        Example
        ====
        ```python
        Chapter34_2().note()
        ```
        """
        print('chapter34.2 note as follow')
        print('34.2 多项式时间的验证')
        print('对语言成员进行“验证”的算法.假如对判定问题PATH的一个给定实例<G,u,v,k>,同时也给定了一条从u到v的路径p.',
            '可以检查p的长度是否至多为k.如果是的,就可以把p看作是该实例的确属于PATH的“证书”.对于判定问题PATH来说,',
            '这一证书没有使我们得益多少.事实上,PATH可以在线性时间内求解')
        print('因此,根据指定的整数来验证成员所需的时间与从头开始解决问题的时间一样长')
        print('哈密顿回路')
        print('在无向图找出哈密顿回路这一问题已经被研究100多年了.形式地说,无向图G=(V,E)中的一个哈密顿回路是通过V中每个顶点一次的简单回路',
            '具有这种回路的图称为哈密顿图,否则称为非哈密顿图')
        print('定义哈密顿回路问题的形式语言：“图G中是否具有一条哈密顿回路”')
        print('HAM-CYCLE={<G>:G是一个哈密顿图}')
        print('已知一个哈密顿回路问题<G>,一种可能的判定算法就是罗列出G的顶点的所有排列,然后对每一种排列进行检查,以确定它是否是一条哈密顿回路',
            '如果我们“合理地”把图编码为邻接矩阵,图中顶点数m为Ω(sqrt(n)),其中n=|<G>|是G的编码长度,则总共会有m!种可能的顶点排列,',
            '因此,算法的运行时间为Ω(m!)=Ω(sqrt(n)!)=Ω(2^sqrt(n)),它并非是O(n^k)的形式(k为任意常数).',
            '因此这种朴素算法的运行时间并不是多项式时间.事实上,哈密顿问题是NP完全的问题')
        print('验证算法')
        print('  假设某给定图G是哈密顿图,并提出可以通过给出沿哈密段回路排列的顶点来证明他的话.证明当然是非常容易的:',
            '仅仅需要检查所提供的回路是否是V中顶点的一个排列,以及沿回路的每条连接的边是否在图中存在,这样就可以验证所提供的回路是否是哈密顿回路',
            '当然,该验证算法可以在O(n^2)的时间内实现,其中n是G的编码的长度',
            '因此,我们可以在多项式时间内验证图中存在一条哈密顿回路和证明过程')
        print('  例如,在哈密顿回路问题中,证书是哈密顿中顶点的列表.如果一个图是哈密顿图,哈密顿回路本身就提供了足够的信息来验证这一事实',
            '相反地,如果某个图不是哈密顿图,那么也不存在这样的顶点列表能使验证算法认为该图是哈密顿图',
            '因为验证算法会会仔细地检查所提供的“回路”是否不是哈密顿回路')
        print('复杂类NP')
        print('  复杂类NP是能被一个多项式时间算法验证的语言类.更准确地说,一个语言L属于NP,当且仅当存在一两个输入的多项式算法A和常数c满足:',
            'L={x∈{0,1}*:存在一个证书y(|y|=O(|x|^c))满足A(x,y)=1}','说算法A在多项式时间内验证了语言L')
        print('  根据先前对哈密顿回路问题的讨论,可知HAM-CYCLE∈NP.此外,如果L∈P,则L∈NP.',
            '如果存在一个多项式时间的算法来判定L,那么只要忽略任何证书,并接受那些它确定属于L的输入串,就可以很容易地把该算法转化为一个两参数的验证算法.因此P∈NP')
        print('目前还不知道是否有P=NP,但大多数研究人员认为P和NP不是同一个类')
        print('从直觉上看,类P由可以很快解决的问题组成,而类NP由可以很快验证其解的问题组成.')
        print('在P!=NP问题之外,还有许多其他基本问题没有解决,尽管很多研究人员做了大量的工作,但还没有人知道NP类在补运算下是否是封闭的',
            '亦即,L∈NP是否说明了L∈NP的语言L的集合')
        print('可以定义复杂类co-NP为满足L∈NP的语言L构成的集合.这样一来,NP在补运算下是否封闭的问题就可以重新表示为是否有NP=co-NP')
        print('练习34.2-1 考虑语言GRAPH-ISOMOPRHISM={<G1,G2>:G1和G2是同构图}.通过描述一个可以在多项式时间内验证该语言的算法',
            '来证明GRAPH-ISOMOPRHISM∈NP')
        print('练习34.2-2 证明:如果G是一个无向的二分图,且有着奇数个顶点,则G是非哈密顿图')
        print('练习34.2-3 证明:如果HAM-CYCLE∈P,则按序列出一个哈密顿回路中的各个顶点的问题是多项式时间可解的')
        print('练习34.2-4 证明:由语言构成的NP类在并集、交集、并置和Kleene星运算下是封闭的')
        print('练习34.2-5 证明:NP中的任何语言都可以用一个运行时间为2^O(n^k)(其中k为常数)的算法来加以判定')
        print('练习34.2-6 图中的哈密顿路径是一种简单路径,经过图中每个顶点一次.证明:语言HAM-PATH={<G,u,v>:图G中存在}')
        print('练习34.2-7 证明：在有向无回路中,哈密顿路径问题可以在多项式时间内求解.给出解决该问题的一个有效算法')
        print('练习34.2-8 设p为一个布尔公式,它由布尔输入变量x1,x2,...,xk,非(~)、AND(∧)、OR(∨)和括号组成.',
            '如果对公式p的输入变量的每一种1和0赋值,公式的结果都为1,则称其为重言式.',
            '定义TAUTOLOGY为由重言布尔公式所组成的语言.证明:TAUTOLOGY∈co-NP')
        print('练习34.2-9 证明P∈co-NP')
        print('练习34.2-10 证明:如果NP!=co-NP,则P!=NP')
        print('练习34.2-11 设G为一个包含至少3个顶点的连通无向图,并设对G中所有由长度至多为3的路径连接起来的点对',
            '将它们直接连接后所形成的图为G^3.证明:G^3是一个哈密顿图.(提示:为G构造一棵生成树,并采用归纳法进行证明)')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_3:
    """
    chapter34.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.3 note

        Example
        ====
        ```python
        Chapter34_3().note()
        ```
        """
        print('chapter34.3 note as follow')
        print('34.3 NP完全性与可归约性')
        print('从事理论研究的计算机科学家们之所以会相信P!=NP,最令人信服的理由可能就是存在着一类“NP完全”问题,该类问题有一种令人惊奇的性质,',
            '即如果一个NP完全问题能在多项式时间内得到解决,那么,NP中的每一个问题都可以在多项式时间内求解')
        print('即P=NP.但是,尽管进行了多年的研究,目前还没有找出关于任何NP完全性问题的多项式时间的算法')
        print('语言HAM-CYCLE就是一个NP完全问题.如果能够在多项式时间内判定HAM-CYCLE,就能够在多项式时间求解NP中的每一个问题.',
            '事实上,如果能证明NP-P为非空集合,就可以肯定地说HAM-CYCLE∈NP-P')
        print('在某种意义上说,NP完全语言是NP中“最难”的语言.在本节中,要说明如何运用称为“多项式时间可归约性”的确切概念,来比较各种语言的相对“难度”')
        print('一个问题Q可以被归约为另一个问题Q\',如果Q的任何实例都可以被“容易地重新描述为Q\'的实例”')
        print('例如,求解关于未知量x的线性方程问题可以转化为求解二次方程问题.已知一个实例ax+b=0',
            '可以把它变换为0x^2+ax+b=0,其解也是方程ax+b=0的解.因此,如果一个问题Q可以转化为另一个问题Q\'',
            '则从某种意义上来说,Q并不比Q\'更难解决')
        print('关于判定问题的形式语言体系,说语言L1在多项式时间内可以归约为语言L2,写作L1<=pL2,',
            '如果存在一个多项式时间可计算的函数f:{0,1}*->{0,1}*,满足对所有的x∈{0,1}*,都有:',
            'x∈L1当且仅当f(x)∈L2.称函数f为归约函数,计算f的多项式时间算法F称为归约算法')
        print('关于从语言L1到另一种语言L2的多项式时间归约的思想.每一种语言都是{0,1}*的子集,归约函数f提供了一个多项式时间的映射,',
            '使得如果x∈L1,则f(x)=L2.如果x∉L1,则f(x)∉L2.因此,归约函数提供了从语言L1表示的判定问题的任意实例x到语言L2表示的判定问题的实例f(x)上映射',
            '如果能提供是否有f(x)∈L2的答案,也就直接提供了是否有x∈L1的答案')
        print('引理34.3 如果L1、L2∈{0,1}*是满足L1<=pL2的语言,则L2∈P蕴含着L1∈P.')
        print('NP完全性')
        print('多项式时间归约提供了一种形式方法,用来证明一个问题在一个多项式时间因子内至少与另一个问题一样难.',
            '亦即,如果L1<=pL2,则L1大于L2的难度不会超过一个多项式时间因子,这就是采用“小于或等于”来表示归约记号的原因')
        print('NP完全语言集合的定义')
        print('语言L∈{0,1}*是NP完全的,如果')
        print('1.L∈NP')
        print('2.对每一个L‘∈NP,有L‘<=pL')
        print('如果一种语言L满足性质2,但不一定满足性质1,则称L是NP难度(NP-hard)的,定义NPC为NP完全语言类')
        print('定理34.4 (NP完全性是判定P是否等于NP的关键) 如果任何NP完全问题是多项式时间可求解的,则P=NP.',
            '等价地,如果NP中的任何问题不是多项式时间可求解的,则所有NP完全问题都不是多项式可求解的')
        print('电路可满足性问题')
        print('  一旦证明了至少有一个问题是NP完全问题,就可以用多项式时间可归约性作为工具,来证明其他问题也具有NP完全性.',
            '下面证明一个NP完全问题:电路可满足性问题')
        print('  布尔组合电路是由布尔组合元素通过电路互连后构造而成的.布尔组合元素是指任何一种电路元素,有着固定数目的输入和输出',
            '执行的是某种良定义的函数功能.布尔值取自集合{0,1},其中0代表FALSE(假),1代表TRUE(真)')
        print('  在电路可满足性问题中,所用到的布尔组合元素计算的是一个简单的布尔函数,这些元素称为逻辑门(组合逻辑电路)')
        print('  三种基本的逻辑门:NOT门(非门,反相器),AND门(与门),OR门(或门).NOT门只有一个二进制输入x,它的值为0或1,产生的是二进制输出z',
            '其值与输入值相反.另外两种门都取两个二进制输入x和y,产生一个二进制输出z')
        print('  用∧来表示AND函数,用∨来表示OR函数.例如0∨1=1,实际的硬件设计中,布尔组合电路可以有多个输出')
        print('  布尔组合电路不包含回路.假设创建了一个有向图G=(V,E),其中每个顶点代表一个组合元素,k条有向边代表每一根扇出为k的接线;',
            '如果某一接线将一个元素u的输出与另一个元素v的输入连接了起来,图中就会有一条有向边(u,v).那么G必定是无回路的')
        print('  电路可满足性问题定义:给定一个由与、或和非门构成的一个布尔组合电路,它是可满足电路吗?',
            '为了给出这一问题的形式定义,必须对电路的编码有一个统一的标准.布尔组合电路的规模是指其中布尔组合元素的个数,',
            '再加上电路中接线的数目.可以设计出一种像图形那样的编码,使其可以把任何给定电路C映射为一个二进制串<C>,',
            '该串的长度与电路本身的规模呈多项式关系.于是定义CIRCUIT-SAT={<C>:C是一个可满足的布尔组合电路}')
        print('  电路可满足性问题在计算机辅助硬件优化领域中极其重要.如果一个子电路总是输出0,就可以用一个更为简单的子电路来取代原电路',
            '该子电路省略了所有的逻辑门,并提供常数值0作为其输出.如果能够开发关于该问题的多项式时间算法,那将具有很大的实际应用价值')
        print('  给定一个电路C,通过检查输入的所有可能赋值来确定它是否是可满足性电路,但如果有k个输入,就会有2^k种可能的赋值.',
            '当电路C的规模为k的多项式时,对每个电路进行检查需要Ω(2^k)的时间,这与电路的规模呈超多项式关系')
        print('  有很强的证据表明:不存在能解决电路可满足性问题的多项式时间算法,因为该问题是NP完全的.根据NP完全性定义中的两个部分,把对这一事实的证明过程也分为两部分')
        print('引理34.5 电路可满足性问题属于NP类')
        print('引理34.6 电路可满足性问题是NP难度的')
        print('定理34.7 电路可满足性问题是NP完全的')
        print('练习34.3-1 略')
        print('练习34.3-2 证明: <=p关系时语言上的一种传递关系.亦即,要证明如果有L1<=pL2,且L2<=pL3,则有L1<=pL3')
        print('练习34.3-3 证明：L<=pL’,当且仅当L’<=pL')
        print('练习34.3-4 证明：在对引理34.5的另一种证明中,可满足性赋值可以当做证书来使用.哪一个证书可以使证明过程更容易些')
        print('练习34.3-5 在引理34.6的证明中,假定算法A的工作存储占用的是一片具有多项式大小的连续存储空间.',
            '在该证明中的什么地方用到了这一假设?论证这一假设是不失一般性的')
        print('练习34.3-6 相对于多项式时间的归约来说,一个语言L对语言类C是完全的,如果对所有L‘∈C,有L∈C.证明:相对于多项式时间的归约来说,',
            'phi和{0,1}*是P中仅有的对P不完全的语言.')
        print('练习34.3-7 证明：L对NP是完全的,当且仅当L’对co-NP是完全的')
        print('练习34.3-8 归约算法F基于有关x、A和k的信息,构造了电路C=f(x).',
            'Sartre教授观察到串x是F的输入,但只有A、k的存在性和运行时间O(n^k)中所隐含的常数因子对F来说是已知的(因为语言L属于NP)',
            '但实际值对F来说却是未知的','因此,这位教授就得出了这样的结论,即F不可能构造出电路C,且语言CIRCUIT-SAT不一定是NP难度的')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_4:
    """
    chapter34.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.4 note

        Example
        ====
        ```python
        Chapter34_4().note()
        ```
        """
        print('chapter34.4 note as follow')
        print('34.4 NP完全性的证明')
        print('电路可满足性问题的NP完全性依赖于直接证明对每一种语言L∈NP,有L<=pCIRCUIT-SAT.本节中将说明,',
            '如何在不把NP中的每一种语言直接归约为给定语言的前提下,证明一种语言是NP完全的.')
        print('引理34.8 如果L是一种满足对某个L’∈NPC,有L’<=pL,则L是NP难度的.此外,如果L∈NP,则L∈NPC')
        print('换句话说,通过把一个已知为NP完全的语言L’归约为L,就可以把NP中的每一种语言都隐式地归约为L.')
        print(' 1) 证明L∈NP')
        print(' 2) 选取一个已知的NP完全语言L’')
        print(' 3) 描述一种算法来计算一个函数f,把L’中的每个实例x∈{0,1}*都映射为L中的一个实例f(x)')
        print(' 4) 证明对所有x∈{0,1}*,函数f满足x∈L’当且仅当f(x)∈L')
        print(' 5) 证明计算函数f的算法具有多项式运行时间')
        print('公式可满足性')
        print('  对于确定布尔公式(而不是电路)是否可满足这一问题,通过给出一个NP完全性证明,来说明上面提到的归约方法.在算法历史上,',
            '这个问题是第一个被证明为NP完全的')
        print('  (公式)可满足性问题可以根据语言SAT描述如下:SAT的一个实例就是一个由下列成分组成的布尔公式p')
        print('  1.n个布尔变量:x1,x2,...,xn')
        print('  2.m个布尔连接词：布尔连接词是任何具有一个或两个输入和一个输出的布尔函数,',
            '如∧(与),∨(或),~(非),->(蕴含),<->(当且仅当)')
        print('  3.括号.(不失一般性,假定没有冗余的括号;亦即,每个布尔连接词至多有一对括号)')
        print('  很容易对一个布尔公式p进行编码,使其长度为n+m的多项式.如在布尔组合电路中一样,环宇一个布尔公式p的真值赋值是为p中各变量所取的一组值;',
            '可满足性赋值是指使公式p的值为1的真值赋值.具有可满足性赋值的董事就是可满足公式.形式语言:',
            'SAT={<p>:p是一个可满足布尔公式}')
        print('  例如公式((x1->x2)∨~((~x1<->x3)∨x4))∧~x2,具有可满足性赋值<x1=0,x2=0,x3=1,x4=1>',
            '这是因为((0->0)∨~((~0<->1)∨1))∧~0=(1∨~(1∨1))∧1=(1∨0)∧1=1,因此,该公式p属于SAT')
        print('  确定一个任意的布尔公式是否是可满足的朴素算法不具有多项式运行时间.在一个具有n个变量的公式p中,有2^n种可能的赋值.',
            '如果<p>的长度是关于n的多项式,则检查每一种可能的赋值需要Ω(2^n)时间,这是<p>长度的一个超多项式')
        print('定理34.9 布尔公式的可满足性问题是NP完全的')
        print('  证明:首先论证SAT∈NP,然后通过证明CIRCUIT-SAT<=pSAT,来证明CIRCUIT-SAT是NP难度的;根据引理34.8可知,这将证明定理成立')
        print('3-CNF可满足性')
        print('  根据公式可满足性进行归约,可以证明很多问题是NP完全问题.归约算法必须能够处理任何输入公式,但这样一来,就必须考虑大量的情况.',
            '因此,常常需要根据布尔公式的一种限制性语言来进行归约,使需要考虑的情况较少')
        print('  当然,不能由于对该语言的限制过多,而使其成为多项式时间可解的语言.3-CNF可满足性(或3-CNF-SAT)就是这样一种方便的语言.')
        print('运用下列术语来定义3-CNF可满足性.布尔公式中的一个文字(literal)是指一个变量或变量的“非”.',
            '如果一个布尔公式可以表示为所有子句(clause)的“与”,且每个子句都是一个或多个文字的“或”,',
            '则称该布尔公式为合取范式,或CNF(conjunctive normal form).如果公式中每个子句恰好都有三个不同的文字,则称该布尔公式为3合取范式,或3-CNF')
        print('例如,布尔公式(x1∨~x1∨~x2)∧(x3∨x2∨x4)∧(~x1∨~x3∨~x4)就是一个3合取范式,其三个子句中的第一个为(x1∨~x1∨~x2),它包含三个文字x1,~x1和~x2')
        print('在3-CNF-SAT中,有这样的问题:3-CNF形式的一个给定布尔公式p是否满足?下列定理说明,即便当布尔公式表述为这种简单范式时,也不可能在多项式时间的算法以确定其满足性')
        print('定理34.10 3合取范式形式的布尔公式的可满足性问题是NP完全的')
        print('归约算法可以分为三个基本步骤.每一步骤都逐渐使输入公式p向所要求的3合取范式接近')
        print('第一步类似于在定理34.9中用于证明CIRCUIT-SAT<=pSAT的过程.首先,为输入公式p构造一棵二叉树“语法分析”树,连接词作为内部顶点:',
            'p=(x1->x2)∨~(~(x1<->x3)∨x4))∧~x2的一棵语法分析树.如果输入公式中有包含数个文字的“或”的子句,就可以利用结合律对表达式加上括号,',
            '以使在所产生的树中的每一个内部顶点上均有1个或两个子女.现在,就可以吧二叉语法分析树看作是计算该函数的一个电路')
        print('归约的第二步是把每个子句p‘i变换为合取范式.通过对p’i中变量的所有可能的赋值进行计算,可以构造出p’i的真值表.',
            '真值表中的每一行由子句变量的一种可能的赋值和根据这一赋值所计算出来的子句的值所组成.',
            '如果运用真值表中值为0的项,就可以构造出公式的析取范式(dissjunctive normal form, DNF),就是“与”的“或”',
            '它等价于~p‘i.然后运用摩根定律并把“或”变成“与”、“与”变成“或”,就可以把公式变换为CNF公式p‘’i')
        print('归约的第三步(也是最后一步)就是继续对公式进行变换,使每个子句恰好有三个不同的文字.',
            '最后的3-CNF公式p’‘’是根据CNF公式p’‘的子句构造出来的,其中使用了两个辅助变量p和q.对p’‘中的每个子句Ci,使p’‘’中包含下列子句')
        print('练习34.4-1 考虑一下在定理34.9的证明中的直接(非多项式时间)归约.描述一个规模为n的电路,当用这种归约方法将其转换为一个公式时,',
            '能产生出一个规模为n的指数的公式') 
        print('练习34.4-2 给出将定理34.10中的方法用于公式(34.3)时所得到的3-CNF公式')
        print('练习34.4-3 Jagger教授提出,在定理34.10的证明中,可以通过仅利用真值表技术(无需其他步骤),就能证明SAT<=p3-CNF-SAT.',
            '亦即,这位教授的意思是取布尔公式p,形成有关其变量的真值表,根据该真值表导出一个3-DNF形式的、等价于~p的公式,',
            '再对其取反,并运用摩根定律,从而可以得到一个等价于p的3-CNF公式.证明:这一策略不能产生多项式时间的归约')
        print('练习34.4-4 证明：确定某一布尔公式是否是重言式这一问题对co-NP来说是完备的.')
        print('练习34.4-5 证明：确定析取范式形式的布尔公式的可满足性这一问题是多项式时间可解的')
        print('练习34.4-6 假设某人给出了一个判定公式可满足性的多项式时间算法.请说明如何利用这一算法在多项式时间内找出可满足性赋值')
        print('练习34.4-7 设2-CNF-SAT为CNF形式的、每个子句中恰有两个文字的可满足公式的集合.证明:2-CNF-SAT∈P.所给出的算法应尽可能地高效.',
            '(提示:注意x∨y与~x->y是等价的.将2-CNF-SAT归约为一个在有向图上高效可解的问题)')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_5:
    """
    chapter34.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.5 note

        Example
        ====
        ```python
        Chapter34_5().note()
        ```
        """
        print('chapter34.5 note as follow')
        print('34.5 NP完全问题')
        print('NP完全问题产生于各种不同领域:布尔逻辑,图论,算术,网络设计,集合与划分,存储于检索,排序与调度,数学程序设计,代数与数论,游戏与趣味难题',
            '自动机与语言理论,程序优化,生物学,化学,物理等等.本节将运用归约方法,对从图论到集合划分的各种问题进行NP完全证明')
        print('图中的每种语言的NP完全性都是根据对指向它的语言进行归约而证明的.其根为CIRCUIT-SAT,在定理34.7中已经证明了它是NP完全语言')
        print('34.5.1 团问题')
        print('无向图G=(V,E)中的团(或团集)(clique)是一个顶点子集V\'∈V,其中每一对顶点之间都由E中的一条边相连.换句话说,一个团是G的一个完全子图.',
            '团的规模是指它所包含的顶点数.团问题是关于寻找图中规模最大的团的最优化问题.形式定义为：',
            'CLIQUE={<G,k>:G是具有规模为k的团的图}')
        print('要确定具有|V|个顶点的图G=(V,E)是否包含一个规模为k的团,一种朴素的算法是列出V的所有k子集,并对其中的每一个进行检查',
            '看它是否形成了一个团.该算法的运行时间为Ω(k^2(|V|,k)),如果k为常数,则它是多项式时间.',
            '但是,在一般情况下,k可能接近于|V|/2,这样一来,算法的运行时间就是超多项式时间.因此,猜想不大可能存在关于团问题的有效算法')
        print('定理34.11 团问题是NP完全的')
        print('  证明：为了证明CLIQUE∈NP,对给定图G=(V,E),用团中顶点集V\'∈V作为G的证书.',
            '对每一对顶点u,v∈V‘,通过检查边(u,v)是否属于E,就可以在多项式时间内,完成对V\'是否是团的检查')
        print('因此,在这个可满足性赋值中,可以将x1设置为0或1')
        print('34.5.2 顶点覆盖问题')
        print('无向图G=(V,E)的顶点覆盖(vertex cover)是指子集V\'∈V,满足如果(u,v)∈E,则u∈V’或v∈V’(或两者成立).',
            '亦即,每个顶点“覆盖”与其关联的边.G的顶点覆盖是覆盖E中所有的边的顶点组成的集合.顶点覆盖的规模是它所包含的顶点数目')
        print('顶点覆盖问题(vertex cover problem)是指在给定的图中,找出具有最小规模的顶点覆盖.',
            '把这一最优化问题重新表述为判定问题,即确定一个图是否具有一个给定规模k的顶点覆盖.作为一种语言,定义',
            'VERTEX-COVER={<G,k>:图G具有规模为k的顶点覆盖}')
        print('定理34.12 顶点覆盖问题是NP完全的')
        print('证明:先来证明VERTEX-COVER∈NP,假定一个图G=(V,E)和整数k.选取的证书是顶点覆盖V\'∈V自身.验证算法证实|V\'|=k,',
            '然后对每条边(u,v)∈E,检查是否有u∈V\',v∈V\'.这一验证可以简单地在多项式时间内进行')
        print('34.5.3 哈密顿回路问题')
        print('定理34.13 哈密顿回路问题是NP完全问题')
        print('  证明：说明HAM-CYCLE属于NP,已知一个图G=(V,E),选取的证书是形成哈密顿回路的|V|个顶点组成的序列',
            '验证算法检查这一序列恰好包含V中每个顶点一次(只有第一个顶点在末尾重复出现一次),并且它们在G中形成一个回路.',
            '亦即,它要检查每一对连续顶点及首、尾顶点之间是否都存在着一条边.该验证算法可以在多项式时间内执行')
        print('34.5.4 旅行商问题')
        print('旅行商问题与哈密顿问题有着密切的联系.在该问题中,一个售货员必须访问n个城市.如果把该问题模型化为一个具有n个顶点的完全图',
            '就可以说这个售货员希望进行一次巡回旅行,或经过哈密顿回路,恰好访问每个城市一次,并最终回到出发的城市.',
            '从城市i到城市j的旅行费用为一个整数c(i,j),这个售货员希望使整个旅行的费用最低,而所需的全部费用是他旅行经过的各边费用之和')
        print('TSP={<G,c,k>:G=(V,E)是一个完全图,c是V*V->Z上的一个函数,k∈Z且包含一个费用至多为k的旅行商的旅行回路}')
        print('定理34.14 旅行商问题是NP完全的')
        print('  证明:首先来说明TSP属于NP.给定该问题的一个实例,用回路中的n个顶点组成的序列作为证书.验证算法检查该序列是否恰好包含每个顶点一次',
            '并且对边的费用求和后,检查和是否至多为k.当然,可以在多项式时间内完成这一过程')
        print('34.5.5 子集和问题')
        print('在子集和问题中,已知一个有限集合S∈N和一个目标t∈N,问题是是否存在一个子集S‘∈S,其元素和为t')
        print('例如,如果S={1,2,7,14,49,98,343,686,2409,2793,16808,17206,117705,117993},t=138457,',
            '则子集S‘={1,2,7,98,343,686,2409,17206,117705}')
        print('SUBSET-SUM={<S,t>:存在一个子集S‘∈S,满足t=∑s}')
        print('与任何算术问题一样,重要的是记住在标准编码,假定输入的整数都是以二进制形式编码的.',
            '在这个假设下,可以证明对于子集和问题,不太可能存在一种快速的算法')
        print('定理34.15 子集和问题是NP完全的')
        print('证明：为了说明SUBSET-SUM属于NP,对该问题的实例<S,t>,设子集S‘是证书.',
            '利用某一验证算法,就可以在多项式时间内完成是否有t=∑s的检查')
        print('练习34.5-1 子图同构问题取两个图G1和G2,要回答G1是否与G2的一个子图同构这一问题.证明:子图同构是NP完全的')
        print('练习34.5-2 给定一个m*n的矩阵A和一个整型的m维向量b,0-1整数规划问题即是否有一个整型的n维变量x,其元素取自集合{0,1}',
            '满足Ax<=b.证明：0-1整数规划问题是NP完全的.')
        print('练习34.5-3 整数线性规划问题与0-1规划问题类似,只是向量x的值可以取自任何整数,而不仅是0或1.假定0-1整数规划问题是NP难度的,',
            '证明整数线性规划问题是NP完全的')
        print('练习34.5-4 证明：如果目标值t表示成一元形式,则子集和问题在多项式时间内可解')
        print('练习34.5-5 集合划分问题的输入为一个数字集合.问题是这些数字是否能被划分成两个集合A和A’,使得∑x(x属于A)=∑x(x属于A’)')
        print('练习34.5-6 证明：哈密顿回路路径问题是NP完全的')
        print('练习34.5-7 最长简单回路问题是在一个图中,找出一个具有最大长度的简单回路(即没有重复的顶点).证明：这个问题是NP完全的')
        print('练习34.5-8 在半3-CNF可满足性,给定一个3-CNF形式的公式p.它包含n个变量和m个子句,其中m是偶数.',
            '希望确定是否存在着对p中变量的一个真值赋值,使得恰有一半的子句为0,恰有一半的子句为1.证明：半3-CNF可满足性问题是NP完全的')
        print('思考题34-1 独立集')
        print('  图G=(V,E)的独立集是子集V∈V’,使得E中的每条边至多与V’中的一个顶点相关联.独立集问题是要找出G中具有最大规模的独立集')
        print('  a) 给出与独立集问题相关的判定问题的形式描述,并证明它是NP完全的.(提示：根据团问题进行归约)')
        print('  b) 假设有一个“黑箱”子程序,用于解决(a)中定义的判定问题.试写出一个算法,以找出规模最大的独立集.',
            '所给出的算法的运行时间应该是关于|V|和|E|的多项式,其中查询黑箱的工作被看作是一步操作')
        print('  c) 当G中的每个顶点的度数均为2时,试写出一个有效的算法来求解独立集问题.分析算法的运行时间,并证明算法的正确性')
        print('  d) 当G为二分图时,试写出一个有效的算法以求解独立集问题.分析算法的运行时间,并证明算法的正确性')
        print('思考题34-2 Bonnie和Clyde')
        print('  Bonnie和Clyde刚刚抢劫了一家银行.他们抢劫到了一袋钱,并打算将钱分光.对于下面的每一种场景,',
            '给出一个多项式时间的算法,或者证明该问题是NP完全的.每一种情况下的输入是关于袋子里n件东西的一份清单,以及每一件东西的价值')
        print('  a) 共有n个硬币,但只有两种不同的面值:一些面值x美元,一些面值y美元.二人希望平分掉这笔钱')
        print('  b) 共有n个硬币,它们有着任意数量的不同面值:但是每一种面值都是2的非负整数次幂,亦即,可能的面值为1美元、2美元、4美元等等.二人希望平分掉这笔钱')
        print('  c) 共有n张支票,十分巧合的是,这些支票恰好是支付给“Bonnie”和“Clyde”的,二人希望平分掉这笔钱')
        print('  d) 与 c)一样,共有n张支票,但这一次,他俩愿意接受这样的一种支票,但这一次,二人所分得的钱数差距不大于100美元')
        print('思考题34-3 图的着色')
        print('  无向图G=(V,E)的k着色(k-coloring)是一个函数c:V->{1,2,...,k},对每条边(u,v)∈E,有c(u)!=c(v).换句话说,',
            '数1,2,..,k表示k种颜色,并且相邻顶点必须为不同的颜色.图的着色问题就是确定要对某个给定图着色所必需的最少的颜色种类')
        print('  a) 写出一个有效的算法,以找出一个图的2着色(如果存在的话)')
        print('  b) 把图的着色问题描述为一个判定问题.证明:该判定问题在多项式时间内可解,当且仅当图的着色问题在多项式时间内可解')
        print('  c) 设语言3-COLOR是能够进行三着色的图的集合.证明：如果3-COLOR是NP完全语言,则b)中的判定问题是NP完全的')
        print('  d) 论证在对包含文字边的图的任意一个3着色c中,一个变量和它的“非”中恰好有一个被着色为c(TRUE),另一种被着色c(FALSE).',
            '论证对p的任何真值赋值,对仅包含文字边的图都有一种3着色存在')
        print('  e) 论证如果x、y和z中每个顶点均着色为c(TRUE)或c(FALSE),则该附件图是3着色的,当且仅当x、y和z中至少有一个被着色为c(TRUE)')
        print('  f) 完成3-COLOR是NP完全问题的证明')
        print('思考题34-4 带收益和完工期限的调度')
        print('  假设有一台机器和n项任务a1,a2,...,an.每一项任务aj都有着处理时间tj、利润pj和完工期限dj.',
            '这台机器一次只能处理一项任务,而任务aj必须不间断地运行tj个连续的时间单位.如果能赶在期限dj之前完成任务aj,就能获取利润pj,',
            '但是如果是在期限到了之后完成任务的,就没有任何利润了.作为一个最优化问题,已知的是n项任务的处理时间、利润和完工期限,',
            '希望找出一种调度方案,以便完成所有的任务,并能获取最大的利润')
        print('  a) 将这个问题表述为一个判定问题')
        print('  b) 证明:此判定问题是NP完全的')
        print('  c) 给出此判定问题的一个多项式时间算法,假定所有的处理时间都是1到n之间的整数.(提示,采用动态规划)')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

chapter34_1 = Chapter34_1()
chapter34_2 = Chapter34_2()
chapter34_3 = Chapter34_3()
chapter34_4 = Chapter34_4()
chapter34_5 = Chapter34_5()

def printchapter34note():
    """
    print chapter34 note.
    """
    print('Run main : single chapter thirty-four!')
    chapter34_1.note()
    chapter34_2.note()
    chapter34_3.note()
    chapter34_4.note()
    chapter34_5.note()

# python src/chapter34/chapter34note.py
# python3 src/chapter34/chapter34note.py

if __name__ == '__main__':  
    printchapter34note()
else:
    pass

```

```py
# coding:utf-8
# usr/bin/python3
# python src/chapter35/chapter35note.py
# python3 src/chapter35/chapter35note.py
"""

Class Chapter35_1

Class Chapter35_2

Class Chapter35_3

Class Chapter35_4

Class Chapter35_5

"""
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    pass
else:
    pass

class Chapter35_1:
    """
    chapter35.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.1 note

        Example
        ====
        ```python
        Chapter35_1().note()
        ```
        """
        print('chapter35.1 note as follow')
        print('第35章 近似算法')
        print('许多具有实际意义的问题都是NP完全问题,但都非常重要,所以不能仅因为获得其最优解的过程非常困难就放弃')
        print('解决NP完全问题至少有三种方法:第一,如果实际输入的规模比较小,则用具有指数运行时间的算法来解决问题就很理想了.',
            '第二,或许能将一些重要的、多项式时间可解的特殊情况隔离出来;第三,仍有可能在多项式时间里面找到(最坏情况或平均情况)近似最优解.',
            '在实践中,近似最优解常常就足够好了,就返回近似最优解的算法称为近似算法')
        print('近似算法的性能比值')
        print('假定在解一个最优化问题,该问题的每一个可能解都有正的代价,希望找出一个近似最优解.根据所要解决的问题',
            '最优解可以定义成具有最大可能代价的解或具有最小可能代价的解.就是说,该问题可能是一个求最大值的问题或求最小值的问题')
        print('说问题的一个近似算法有着近似比p(n),如果对规模为n的任何输入,由该近似算法产生的解的代价C与最优解的代价C*只差一个因子p(n)',
            'max(C/C*,C*/C)<=p(n),也称一个能达到近似比p(n)的算法为p(n)近似算法.这个定义对求最大值和求最小值问题都适用.',
            '对于一个求最大值的问题,0<C<=C*,而比值C*/C给出最优解的代价大于近似解的代价的倍数.类似地,对于求最小值问题也是同理')
        print('对于很多问题来说,已经设计出具有较小的固定近似比的多项式时间近似算法;对于另一些问题来说,在其已知的最佳多项式时间的近似算法中,',
            '近似比是作为输入规模n的函数而增长的')
        print('一些NP完全问题允许有多项式时间的近似算法,通过消耗越来越多的计算时间,这些近似算法可以达到不断缩小的近似比.就是说,在计算时间和近似的质量之间可以进行权衡')
        print('一个最优化问题的近似方案是这样的一种近似算法,它的输入除了该问题的实例外,还有一个值e>0,使得对任何固定的e,该方案是个(1+e)近似算法',
            '对一个近似方案来说,如果对任何固定的e>0,该方案都以其输入实例的规模n的多项式时间运行,则称此方案为多项式时间近似方案')
        print('随着e的减小,多项式时间近似方案的运行时间会迅速增长.例如,一个多项式时间近似方案的运行时间可能达到O(n^2/e).在理想情况下,',
            '如果e按一个常数因子减小,为了获得希望的近似效果,所增加的运行时间不应该超过一个常数因子.',
            '希望运行时间既是1/e的多项式,又是n的多项式')
        print('对一个近似方案来说,如果其运行时间既是1/e的多项式,又为输入实例的规模n的多项式,则称其为完全多项式时间的近似方案.',
            '例如,近似方案可能有O((1/e)^2n^3)运行时间.对于这样的一种方案,e的任意常数倍的减少可以由运行时间的相应常数倍增加来弥补')
        print('35.1 顶点覆盖问题')
        print('虽然在一个图G中寻找最优顶点覆盖比较困难,但要找出一个近似最优的顶点覆盖不会太难.下面给出的近似算法以一个无向图G为输入,',
            '并返回一个其规模保证不超过最优顶点覆盖的规模两倍的顶点覆盖规模两倍的顶点覆盖')
        print('定理35.1 APPROX-VERTEX-COVER有一个多项式时间的2近似算法')
        print('练习35.1-1 给出一个图的例子,使得APPROX-VERTEX-COVER对该图总是产生次最优解')
        print('练习35.1-2 设A表示在APPROX-VERTEX-COVER的第4行中挑选出来的边集.证明:集合A是图G中的一个最大匹配')
        print('练习35.1-3 Nixon教授提出了以下的启发式方法来解决顶点覆盖问题:重复地选择度数最高的顶点并去掉所有邻接边.给出一个例子,',
            '说明这位教授的启发式方法达不到近似比2(提出:可以考虑一个二分图,其中左图中的顶点的度数一样,而右图中顶点的度数不一样)')
        print('练习35.1-4 给出一个有效的贪心算法,以便在线性时间内,找出一棵树的最优顶点覆盖')
        print('练习35.1-5 顶点覆盖问题和NP完全团问题在某种意义上来说是互补的,即最优顶点覆盖是补图中某一最大规模团的补.',
            '这种关系是否意味着存在着一个多项式时间的近似算法,它对团问题有着固定的近似比?请给出你的回答,并对你的回答加以说明')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_2:
    """
    chapter35.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.2 note

        Example
        ====
        ```python
        Chapter35_2().note()
        ```
        """
        print('chapter35.2 note as follow')
        print('35.2 旅行商问题')
        print('给定一个完全的无向图G=(V,E),其中每条边(u,v)∈E都有一个非负的整数代价c(u,v),希望找出G的一个具有最小代价的哈密顿回路.')
        print('在很多实际情况中,从一个地方u直接到另一个地方w总是代价最小的.经由任何一个中转站v的一种路径不可能具有更小的代价了.',
            '换句话说,去掉一个中间站绝不会增加代价.将这一概念加以形式化,即如果对所有的顶点u,v,w∈V,有:c(u,w)<=c(u,v)+c(v,w)',
            '就称代价函数c满足三角不等式c(u,w)<=c(u,v)+c(v,w)就称代价函数c满足三角不等式')
        print('三角不等式是个很自然的不等式,在许多应用中,都能自动得到满足.例如:如果图的顶点为平面上的点,',
            '且在两个顶点之间旅行的代价即为它们之间通常的欧几里得距离,就满足三角不等式')
        print('即使强行要求代价函数满足三角不等式,也不能改变旅行商问题的NP完全性.',
            '因此,不可能找出一个准确解决这个问题的多项式时间算法,因而就要寻找一些好的近似算法')
        print('在35.2.1节中要讨论一个2近似算法,用于解决符合三角不等式的旅行商问题.在35.2.2节中,要证明如果不符合三角不等式,',
            '则不存在具有常数近似比多项式时间的近似算法,除非P=NP')
        print('35.2.1 满足三角不等式的旅行商问题')
        print('利用前一小节的方法,首先计算出一个结构(即最小生成树),其权值是最优旅行商路线长度的下界.接着,要利用这一最小生成树来生成一条遍历线路',
            '其代价不大于最小生成树权值的两倍,只要代价函数满足三角不等式即可')
        print('即使是采用MST-PRISM的简单实现,APPROX-TSP-TOUR的运行时间也是Θ(V^2).现在我们来证明：如果旅行商问题的某一实例的代价函数满足三角不等式,',
            'APPROX-TSP-TOUR所返回的游程的代价不大于最优游程的代价的两倍')
        print('定理35.2 APPROX-TSP-TOUR是一个解决满足三角不等式的旅行商问题的、多项式时间的2近似算法')
        print('  证明:前面已经证明了APPROX-TSP-TOUR的运行时间为多项式')
        print('35.2.2 一般旅行商问题')
        print('如果去掉关于代价函数c满足三角不等式的假设,则不可能在多项式时间内找到好的近似路线,除非N=NP')
        print('定理35.3 如果P!=NP则对任何常数p>=1,一般旅行商问题不存在具有近似比p的多项式时间近似算法')
        print('练习35.2-1 假设有一个完全无向图G=(V,E),含有至少三个顶点,其代价函数c满足三角不等式.证明:对所有的u,v∈V,有c(u,v)>=0')
        print('练习35.2-2 说明如何才能在多项式时间内,将旅行商问题的一个实例转换为另一个其代价函数满足三角不等式的实例.',
            '两个实例必须有同一组最优游程.请解释为什么这样的一种多项式时间的转换与定理35.3并不矛盾,假设P!=NP.')
        print('练习35.2-3 考虑以下的用于构造近似旅行商游程的最近点启发式:从只包含任意选择的某一顶点的平凡回路开始.',
            '在每一步中,找出一个顶点u,它不在回路中,但与回路上任何顶点之间的距离最短.假设回路上距离u最近的顶点为v.将回路加以扩展以包含顶点u,即将u插入在v之后',
            '重复这一过程,直到所有的顶点都在回路上时为止.证明：这一启发式返回的游程总代价不大于最优游程代价的两倍')
        print('练习35.2-4 在瓶颈旅行商问题中,要找出这样的一条哈密顿回路,使得回路中代价最大的边的代价最小.假设代价函数满足三角不等式,',
            '证明：这个问题存在着一个近似比为3的多项式时间近似算法(采用递归证明的方法,通过完全遍历瓶颈生成树及跳过某些顶点,可以恰访问书中的每个顶点一次),',
            '但连续跳过的中间顶点不会多余两个.证明在瓶颈生成树中,代价最大的边的代价至多为瓶颈哈密顿葫芦中代价最大的边的代价')
        print('练习35.2-5 假设与旅行商问题的一个实例对应的顶点是平面上的点,且代价c(u,v)是点u和v之间的欧几里得距离.证明:一条最优游程不会自我交叉')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_3:
    """
    chapter35.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.3 note

        Example
        ====
        ```python
        Chapter35_3().note()
        ```
        """
        print('chapter35.3 note as follow')
        print('35.3 集合覆盖问题')
        print('集合覆盖问题是一个最优化问题,它模型化了许多资源选择问题,其相应的判定问题推广了NP完全的顶点覆盖问题,因而也是NP难度的.',
            '然而,用于解决顶点覆盖的近似算法在这儿就用不上了,需要尝试其他的一些方法.','要讨论一种简单的带对数近似比的贪心启发式方法,',
            '亦即,随着实例规模逐渐增大,相对于一个最优解的规模来说,近似解的规模也可能增大.但是,由于对数函数增长很慢,故这个近似算法可能会产生出很有用的结果来',
            '集合覆盖问题的一个实例(X,F)由一个有穷集合X和一个X的子集族F构成,且X的每一个元素属于F中的至少一个子集X=(S∈F)∪S')
        print('集合覆盖问题是对许多常见的组合问题的一种抽象.来看一个简单的例子：假设X表示解决某一问题所需要的各种技巧的集合,另外有一个给定的可用来解决该问题的人的集合.')
        print('一个贪心的近似算法')
        print('贪心方法在每一阶段都选择出能覆盖最多的、未被覆盖的元素的集合S(GREEDY-SET-COVER)')
        print('GREEDY-SET-COVER算法的工作过程是这样的,在每个阶段,集合U包含余下的未被覆盖的元素构成的集合；集合C包含正在被构造的覆盖.',
            '贪心决策步骤：即选出一个子集S,它能覆盖尽可能多的未被覆盖的元素(如果有两个子集覆盖了一样多的元素,可以任意选择其中的一个).',
            '在S被选出后,将其元素从U中去掉,b并将S置于C中.当该算法结束时,集合C就包含了一个覆盖X的F子族')
        print('GREEDY-SET-COVER算法存在一个运行时间为O(|X||F|min(|X|,|F|))的实现.')
        print('下面来证明以上的贪心算法可以返回一个比最优集合覆盖大不了很多的集合覆盖.为方便起见,在本章中,用H(d)来表示第d级调和数Hd=∑1/i',
            '作为一个边界条件,定义H(0)=0')
        print('定理35.4 GREEDY-SET-COVER是一个多项式时间的p(n)近似算法,其中p(n)=H(max{|S|:S∈F})')
        print('证明:已经证明了GREEDY-SET-COVER是以多项式时间运行的')
        print('推论35.5 GREEDY-SET-COVER是一个多项式时间的(ln|X|+1)近似算法')
        print('在某些应用中,max{|S|,S∈F}是一个较小的常数,在这种情况下,由GREEDY-SET-COVER返回的解就至多比最优解大一个很小的常数倍.',
            '例如,对一个其顶点的度至多为3的图来说,当利用这种启发式来获取其近似顶点覆盖时,即会出现这样的应用.',
            '在这种情况下,由GREEDY-SET-COVER找出的解不大于一个最优解的H(3)=11/6倍,这个性能保证比APPROX-VERTEX-COVER的要略好一些')
        print('练习35.3-1 将以下的每一个单词都看作是字母的集合:{arid,dash,drain,heard,lost,nose,shun,slate,snare,thread}.',
            '说明当出现两个候选集合可供选择时,如果倾向于在词典中先出现的单词,则GREEDY-SET-COVER会产生怎样的集合覆盖')
        print('练习35.3-2 通过从顶点覆盖问题进行归约,证明集合覆盖问题的判定版本是NP完全的')
        print('练习35.3-3 说明如何实现GREEDY-SET-COVER,使其运行时间为O(∑|S|)')
        print('练习35.3-4 以下给出的是定理35.4的较弱一些形式,证明它是成立的:|C|<=|C*|max{|S|:S∈F}')
        print('练习35.3-5 GREEDY-SET-COVER可以返回一些不同的解')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_4:
    """
    chapter35.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.4 note

        Example
        ====
        ```python
        Chapter35_4().note()
        ```
        """
        print('chapter35.4 note as follow')
        print('35.4 随机化和线性规划')
        print('给出3-CNF可满足性的最优化版本的一个随机化算法,还要利用线性规划,为顶点覆盖问题的一个带权版本设计近似算法.')
        print('解决MAX-3-CNF可满足性问题的一个随机化近似算法')
        print('正如存在着可以计算出准确解的随机算法一样,也存在着能够计算近似解的随机化算法.',
            '称某一问题的一个随机化算法具有近似比p(n),如果对任何规模为n的输入,该随机化算法所产生的解的期望代价C是在最优解的代价C*的一个因子p(n)之内:')
        print('  max(C/C*,C*/C)<=p(n)')
        print('能达到近似比p(n)的随机化算法也称为随机化的p(n)近似算法.换句话说,随机化的近似算法类似于确定型算法,只是其近似比的一个期望值')
        print('亦即,希望找出变量的一种赋值,使得有尽可能的子句得到满足.称这种最大化问题为MAX-3-CNF-可满足性问题的一种赋值,它能最大化结果为1的子句的数量.')
        print('定理35.6 给定MAX-3-CNF可满足性问题的一个实例,有n个变量x1,x2,...,xn和m个子句,',
            '以概率1/2独立地将每个变量设置为1和以概率1/2独立地将每个变量设置为0的随机化近似算法是一个随机化的8/7近似算法')
        print('利用线性规划来近似带权顶点覆盖')
        print('  在最小权值顶点覆盖问题,给定一个无向图G=(V,E),其中每个顶点v∈V都有一个关联的正的权值w(v).对任意顶点覆盖V\'∈V,',
            '定义该顶点覆盖的权为w(V‘)=∑w(v).目标是找出一个具有最小权值的顶点覆盖')
        print('  假设对每一个顶点v∈V,都安排一个变量x(v)与之关联,并且,要求对每一个顶点v∈V,有x(v)∈[0,1].',
            '将x(v)=1解释为v在顶点覆盖中,将x(v)=0解释为v不在顶点覆盖中.写出这样一条限制,即对于任意边(u,v),u和v之中至少有一个必须在顶点覆盖中',
            '即x(u)+x(v)>=1.这样一来,就引出了以下的用于寻找最小权值顶点覆盖的0-1整数规划')
        print('于是,线性规划的一个最优解是0-1整数规划最优解的一个下界,从而也是最小权值顶点覆盖问题最优解的下界')
        print('APPROX-MIN-WEIGHT-VC过程利用上述线性规划的解,构造最小权值顶点覆盖问题的一个近似解')
        print('定理35.7 APPROX-MIN-WEIGHT-VC是解决最小权值顶点覆盖问题的一个多项式时间的2近似算法')
        print('练习35.4-1 证明:即使允许一个子句既包含变量又包含其否定形式,将每个变量随机地以概率1/2设置为1和以概率1/2设置为0,它仍然是一个随机化的8/7近似算法')
        print('练习35.4-2 MAX-CNF可满足性问题与MAX-3-CNF可满足性问题类似,只是它并不限制每个子句都包含3个问题.',
            '对MAX-CNF可满足性问题,给出它的一个随机化的2近似算法')
        print('练习35.4-3 在MAX-CUT问题中,给定一个无权无向图G=(V,E).定义一个割(S,V-S),并定义一个割的权为通过该割的边的数目.',
            '问题的目标是找出一个具有最大权值的割.假设对每个顶点v,随机地且独立地将v以概率1/2置入S中,以概率1/2置入V-S中.证明:这个算法是一个随机化的2近似算法')
        print('练习35.4-4 略')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_5:
    """
    chapter35.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.5 note

        Example
        ====
        ```python
        Chapter35_5().note()
        ```
        """
        print('chapter35.5 note as follow')
        print('35.5 子集和问题')
        print('子集和问题的一个实例是一个对(S,t),其中S是一个正整数集合{x1,x2,...,xn},t为一个正整数.这个判定问题是判定是否存在S的一个子集,',
            '使得其中的数加起来恰为目标值t.这个问题是NP完全的')
        print('与此判定问题相联系的最优化问题常常出现于实际应用中,在这种最优化问题中,希望找到{x1,x2,...,xn}的一个子集,使其中元素相加之和尽可能的大,但不能大于t',
            '例如,假设有一辆能装不多于t磅重的货的卡车,并有n个不同的盒子要装运,其中第i个的重量为xi磅,希望在不超过重量极限的前提下,将货尽可能地装满卡车.')
        print('先给出解决这个最优化问题的一个指数时间算法,然后说明如何来修改算法,使之成为一个完全多项式时间的近似方案.',
            '(一个完全多项式时间近似方案的运行时间为1/e以及输入规模的多项式)')
        print('一个指数时间的准确算法')
        print('  假设对S的每个子集S\',都计算出S‘中所有元素的和.接着,在所有其元素和不超过t的子集中,选择其和最接近t的那个子集.',
            '显然,这一算法将返回最优解.但它可能需要指数级的时间.为了实现这个算法,可以采用一种迭代过程:在第i轮迭代中,计算{x1,x2,...,xi}的所有子集的元素和,',
            '计算的基础是{x1,x2,...,xi-1}的所有子集的和.在此计算过程中,一旦某个特定的子集S\'的和超过了t,就没有必要再对它进行处理了,因为S\'的超集都不会成为最优解')
        print('  过程EXACT-SUBSET-SUM的输入为一个集合S={x1,x2,...,xn}和一个目标值t;整个过程以迭代的方式计算列表Li,其中列出了{x1,x2,...,xi}的所有子集的和,',
            '这些和值都不超过目标值t.接着,它返回Ln中的最大值')
        print('如果L是一个由正整数所构成的表,x是一个正整数,用L+x来表示通过对L中每个元素增加x而导出的整数列表.例如,如果L=<1,2,3,5,9>,则L+2=<3,4,5,7,11>')
        print('一个完全多项式时间近似方案')
        print('  对于子集和问题,可以导出一个完全多项式时间近似方案,在每个列表Li被创建后,对它进行“修整”.具体的思想是如果L中的两个值比较接近,',
            '那么,处于寻找近似解的目的,没有理由同时保存这两个数.更准确地说,采用一个修整参数d,满足0<d<1,按d来修整一个列表L意味着以这样一种方式从L中除尽可能多的元素,即如果L\'为修整L后的结果,',
            '则对从L中去除的每个元素y,都存在着一个仍在L\'中的,近似y的元素z,使得y/(1+d)<=z<=y')
        print('  例如d=0.1,L=<10,11,12,15,20,21,22,23,24,29>,可以修整L得L\'=<10,12,15,20,23,29>')
        print('定理35.8 APPROX-SUBSET-SUM是关于子集和问题的一个完全多项式时间近似方案')
        print('练习35.1-1 证明在执行了EXACT-SUBSET-SUM的第5行之后,Li是一个有序表,它包含了Pi中所有不大于t的元素')
        print('练习35.1-2 略')
        print('练习35.1-3 略')
        print('练习35.1-4 设t为给定输入列表的某个子集之和,如何找出不小于t的最小值的良好近似')
        print('思考题35-1 装箱')
        print('  假设有一组n个物体,其中第i个物体的大小si,满足0<si<1.希望把所有的物体都装入最少的箱子中,',
            '这些箱子为单位尺寸大小,即每个箱子能容纳所有物体的一个总尺寸不大于1的子集')
        print(' a) 证明:确定最少箱子个数的问题是NP难度的(提示：对子集和问题进行归约,',
            '首先适合启发式依次考察每个物体,将其放入能容纳它的第一个箱子.设S=∑si)')
        print(' b) 论证:所需箱子的最优个数至少为[S],所需箱子的最优个数至少为[S]')
        print(' c) 论证:首先适合启发式至多使一个箱子不到半满')
        print(' d) 证明:由首先适合启发式用到的箱子数绝不会大于[2S]')
        print(' e) 证明:首先适合启发式具有近似比2')
        print(' f) 给出首先适合启发式的一个有效实现,并分析其运行时间')
        print('思考题35-2 对最大团规模的近似')
        print('  设G=(V,E)为一个无向图。对任意k>=1,定义G(k)为无向图(V(k),E(k)),其中V(k)是V中顶点的所有有序k元组构成的集合,',
            'E(k)被定义成(v1,v2,...,vk)与(w1,w2,...,wk)邻接,当且仅当对每一个i(1<=i<=k)G中或者有vi与wi邻接,或者有vi=wi')
        print('  a) 证明：G(k)中最大团的大小等于G中最大团的大小的k次幂')
        print('  b) 论证：如果有一个寻找最大规模团的近似算法,其近似比为常数,则该问题存在一个完全多项式时间的近似方案')
        print('思考题35-3 带权集合覆盖问题')
        print('  假设将集合覆盖问题加以一般化,使得族F中的每个集合Si都有一个权值wi,而一个覆盖C的权则为∑wi.希望确定一个具有最小权值的覆盖.',
            '证明贪心集合覆盖启发式可以很自然的方式加以推广,对带权集合覆盖问题的任何实例提供一个近似解.证明该启发式有一个近似比H(d),其中d为任意集合Si的最大规模')
        print('思考题35-4 最大匹配')
        print('  在一个无向图G中,所谓匹配(matching)是指这样的一组边,其中任意两条边都不关联于同一顶点.',
            '看到了如何在一个二分图中寻找最大匹配.在本问题中,要来考察一般无向图(即不必是二分图的无向图)中的匹配问题')
        print('  a) 极大匹配是指不是任何其他匹配的真子集的匹配.通过给出一个无向图G和G中的极大匹配M(它不是一个最大匹配),',
            '来证明极大匹配未必是最大匹配')
        print('  b) 考虑一个无向图G=(V,E).给出一个O(E)时间的贪心算法,用于寻找G中的极大匹配')
        print('  c) 证明:G的一个最大匹配的规模是G的任何顶点覆盖的规模的下界')
        print('  d) 考虑G=(V,E)中的一个极大匹配M.设T={v∈V,M中的某条边与v关联}.对于由G的那些不在T中的顶点而构成的子图,应如何分析它们')
        print('  e) 根据d)得出这样的结论:2|M|是G的顶点覆盖的规模')
        print('  f) 利用c)和e),证明b)中的贪心算法是有关最大匹配的一个2近似算法')
        print('思考题35-5 并行机调度')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

chapter35_1 = Chapter35_1()
chapter35_2 = Chapter35_2()
chapter35_3 = Chapter35_3()
chapter35_4 = Chapter35_4()
chapter35_5 = Chapter35_5()

def printchapter35note():
    """
    print chapter35 note.
    """
    print('Run main : single chapter thirty-five!')
    chapter35_1.note()
    chapter35_2.note()
    chapter35_3.note()
    chapter35_4.note()
    chapter35_5.note()

# python src/chapter35/chapter35note.py
# python3 src/chapter35/chapter35note.py

if __name__ == '__main__':  
    printchapter35note()
else:
    pass

```

```py

from graph import *
from copy import deepcopy

def approx_vertex_cover(g : Graph):
    """
    顶点覆盖问题的近似算法
    """
    C = []
    edges = deepcopy(g.edges)
    while len(edges) != 0:
        u, v = g.getvertexfromedge(edges[0])
        C += [u.key, v.key]
        i = 0
        while i < len(edges):
            edge = edges[i]
            if edge.vertex1.key == u.key or edge.vertex1.key == v.key or edge.vertex2.key == u.key or edge.vertex2.key == v.key:
                edges.pop(i)
            else:
                i += 1
    return C

def trim(L, d):
    """
    对列表`L`的修整
    """  
    m = len(L)
    L_ = [L[0]]
    last = L[0]
    for i in range(1, m):
        if L[i] > last * (1 + d):
            L_.append(L[i])
            last = L[i]
    return L_

def approx_subset_sum(S, t, e):
    n = len(S)
    l_last = [0]
    for i in range(n):
        l = sorted(l_last + list(map(lambda x : x + S[i], l_last)))
        l = trim(l, e / (2 * n))
        l_last.clear()
        for num in l:
            if num <= t:
                l_last.append(num)
    return max(l_last)

def test_approx_vertex_cover():
    """
    测试顶点覆盖问题的近似算法
    """
    g = Graph()
    g.addvertex(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    g.addedge('a', 'b')
    g.addedge('b', 'c')
    g.addedge('c', 'd')
    g.addedge('c', 'e')
    g.addedge('d', 'e')
    g.addedge('d', 'f')
    g.addedge('d', 'g')
    g.addedge('e', 'f')
    g.printadj()
    print(approx_vertex_cover(g))

def test():
    """
    测试函数
    """
    test_approx_vertex_cover()
    print(trim([10, 11, 12, 15, 20, 21, 22, 23, 24, 29], 0.1))
    print(approx_subset_sum([104, 102, 201, 101], 308, 0.4))
    print(approx_subset_sum([104, 102, 201, 101, 100, 20, 123], 450, 0.4))

if __name__ == '__main__':
    test()
else:
    pass

```

```py

'''
排序算法集合

First
=====

冒泡排序 `O(n^2)`      ok

鸡尾酒排序(双向冒泡排序) `O(n^2)`

插入排序 `O(n^2)`      ok

桶排序 `O(n)`          ok

计数排序 `O(n + k)`     ok

合并排序 `O(nlgn)`      ok

原地合并排序 `O(n^2)`    ok

二叉排序树排序 `O(nlgn)`  ok

鸽巢排序 `O(n+k)`

基数排序 `O(nk)`        ok

Gnome排序 `O(n^2)`

图书馆排序 `O(nlgn)`

Second
======

选择排序 `O(n^2)`    ok

希尔排序 `O(nlgn)`   

组合排序 `O(nlgn)`

堆排序  `O(nlgn)`   ok

平滑排序  `O(nlgn)`

快速排序   `O(nlgn)`

Intro排序  `O(nlgn)`

Patience排序 `O(nlgn + k)`

Third
=====

Bogo排序 `O(n*n!)`

Stupid排序 `O(n^3)`

珠排序  `O(n) or O(sqrt(n))`

Pancake排序   `O(n)`

Stooge排序  `O(n^2.7)`   ok

'''

# python src/dugulib/sort.py
# python3 src/dugulib/sort.py
from __future__ import division, absolute_import, print_function
import math as _math
import random as _random
from copy import deepcopy as _deepcopy
from numpy import arange as _arange

__all__ = ['insertsort', 'selectsort', 'bubblesort',
               'mergesort', 'heapsort', 'quicksort', 
               'stoogesort'].sort()

class Sort:
    '''
    排序算法集合类
    '''
    def insertsort(self, array : list) -> list:
        '''
        Summary
        ===
        插入排序的升序排列,时间复杂度`O(n^2)`
    
        Parameter
        ===
        `array` : a list like
        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> sort.insertsort(array)
        >>> [1, 2, 3, 4, 5, 6]
        ```
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

    def selectsort(self, array : list = []) -> list:
        '''
        Summary
        ===
        选择排序的升序排列,时间复杂度:O(n^2):
        
        Args
        ===
        `array` : a list like

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> array = [1, 3, 5, 2, 4, 6]
        >>> sort.selectsort(array)
        >>> [1, 2, 3, 4, 5, 6]
        ```
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

    def bubblesort(self, array : list) -> list:
        '''
        冒泡排序,时间复杂度`O(n^2)`

        Args
        ====
        `array` : 排序前的数组

        Return
        ======
        `sortedArray` : 使用冒泡排序排好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> sort.bubblesort(A)
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        nums = _deepcopy(array)
        for i in range(len(nums) - 1):    
            for j in range(len(nums) - i - 1):  
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums

    def __mergeSortOne(self, array : list, p : int ,q : int, r : int) -> list:
        '''
        一步合并两堆牌排序算法过程

        Args
        ===
        `array` : a array like

        Returns
        ===
        `sortedArray` : 排序好的数组

        Raises
        ===
        `None`
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
        L = _arange(n1 + 1, dtype=float)
        R = _arange(n2 + 1, dtype=float)
        # 将A分堆
        for i in range(n1):
            L[i] = A[p + i]
        for j in range(n2):
            R[j] = A[q + j + 1]
        # 加入无穷大“哨兵牌”, 对不均匀分堆的完美解决
        L[n1] = _math.inf
        R[n2] = _math.inf
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

    def __mergeSort(self, array : list, start : int, end : int) -> list:
        '''
        合并排序总过程

        Args
        ===
        `array` : 待排序数组
        `start` : 排序起始索引
        `end` : 排序结束索引

        Return
        ===
        `sortedArray` : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> sort.mergeSort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        # python一切皆对象和引用，所以要拷贝...特别是递归调用的时候
        r = _deepcopy(end)
        p = _deepcopy(start)
        if p < r:
            # 待排序序列劈成两半
            middle = (r + p) // 2
            q = _deepcopy(middle)
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

    def mergesort(self, array : list) -> list:
        '''
        归并排序/合并排序：最优排序复杂度`O(n * log2(n))`, 空间复杂度`O(n)`

        Args
        ===
        array : 待排序的数组

        Returns
        ===
        sortedArray : 排序好的数组

        Example
        ===
        ```python
        >>> import sort
        >>> sort.mergesort([6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6]
        ```
        '''
        return self.__mergeSort(array, 0, len(array) - 1)

    def left(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:左儿子:的下标
        '''
        return int(2 * i + 1)

    def right(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:右儿子:的下标
        '''
        return int(2 * i + 2)

    def parent(self, i : int) -> int:
        '''
        求:二叉堆:一个下标i的:父节点:的下标
        '''
        return (i + 1) // 2 - 1

    def heapsize(self, A : list) -> int:
        '''
        求一个数组形式的:二叉堆:的:堆大小:
        '''
        return len(A) - 1

    def maxheapify(self, A : list, i : int) -> list:
        '''
        保持堆使某一个结点i成为最大堆(其子树本身已经为最大堆) :不使用递归算法:
        '''
        count = len(A)
        largest = count
        while largest != i:
            l = self.left(i)
            r = self.right(i)
            if l <= self.heapsize(A) and A[l] >= A[i]:
                largest = l
            else:
                largest = i
            if r <= self.heapsize(A) and A[r] >= A[largest]:
                largest = r
            if largest != i:
                A[i], A[largest] = A[largest], A[i]
                i, largest = largest, count
        return A

    def buildmaxheap(self, A : list) -> list:
        '''
        对一个数组建立最大堆的过程, 时间代价为:O(n):
        '''
        count = int(len(A) // 2)
        for i in range(count + 1):
            self.maxheapify(A, count - i)
        return A

    def heapsort(self, A : list) -> list:
        '''
        堆排序算法过程, 时间代价为:O(nlgn):

        Args
        ===
        A : 待排序的数组A

        Return
        ====
        sortedA : 排序好的数组

        Example
        ====
        ```python
        >>> import sort
        >>> sort.heapsort([7, 6, 5, 4, 3, 2, 1])
        >>> [1, 2, 3, 4, 5, 6, 7]
        ```
        '''
        heapsize = len(A) - 1

        def left(i : int):
            '''
            求:二叉堆:一个下标i的:左儿子:的下标
            '''
            return int(2 * i + 1)

        def right(i : int):
            '''
            求:二叉堆:一个下标i的:右儿子:的下标
            '''
            return int(2 * i + 2)

        def parent(i : int):
            '''
            求:二叉堆:一个下标i的:父节点:的下标
            '''
            return (i + 1) // 2 - 1

        def __maxheapify(A : list, i : int):
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

        self.buildmaxheap(A)
        length = len(A)   
        for i in range(length - 1):
            j = length - 1 - i
            A[0], A[j] = A[j], A[0]
            heapsize = heapsize - 1
            __maxheapify(A, 0)
        return A

    def partition(self, A : list, p : int, r : int):
        '''
        快速排序的数组划分子程序
        '''
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

    def __quicksort(self, A : list, p : int, r : int):
        left = _deepcopy(p)
        right = _deepcopy(r)
        if left < right:
            middle = _deepcopy(self.partition(A, left, right))
            self.__quicksort(A, left, middle - 1)
            self.__quicksort(A, middle + 1, right)

    def quicksort(self, A : list):
        '''
        快速排序，时间复杂度`o(n^2)`,但是期望的平均时间较好`Θ(nlgn)`

        Args
        ====
        `A` : 排序前的数组`(本地排序)`

        Return
        ======
        `A` : 使用快速排序排好的数组`(本地排序)`

        Example
        ===
        ```python
        >>> import sort
        >>> A = [6, 5, 4, 3, 2, 1]
        >>> sort.quicksort(A)
        >>> [1, 2, 3, 4, 5, 6]
        '''
        self.__quicksort(A, 0, len(A) - 1)
        return A

    def __stoogesort(self, A, i, j):
        if A[i] > A[j]:
            A[i], A[j] = A[j], A[i]
        if i + 1 >= j:
            return A
        k = (j - i + 1) // 3
        __stoogesort(A, i, j - k)
        __stoogesort(A, i + k, j)
        return __stoogesort(A, i, j - k)

    def stoogesort(self, A : list) -> list:
        '''
        Stooge原地排序 时间复杂度为:O(n^2.7):

        Args
        ===
        `A` : 排序前的数组:(本地排序):
        '''
        return __stoogesort(A, 0, len(A) - 1)

    def shellsort(self, A : list):
        """
        希尔排序 时间复杂度为:O(nlogn) 原地排序
        """
        n = len(A)
        fraction = n // 2
        while fraction > 0:
            for i in range(fraction, n):
                for j in range(i - fraction, -1, -fraction):
                    if A[j] > A[j + fraction]:
                        A[j], A[j + fraction] = A[j + fraction], A[j]
                    else:
                        break
            fraction //= 2
        return A

    def countingsort2(self, A):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> countingsort2([0,1,1,3,4,6,5,3,5])
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        return self.countingsort(A, max(A) + 1)

    def countingsort(self, A, k):
        '''
        针对数组`A`计数排序，无需比较，非原地排序，当`k=O(n)`时，算法时间复杂度为`Θ(n)`,
        3个n for 循环
        需要预先知道数组元素都不大于`k`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B

    def getarraystr_subarray(self, A, k):
        '''
        取一个数组中每个元素第k位构成的子数组

        Args
        ===
        `A` : 待取子数组的数组

        `k` : 第1位是最低位，第d位是最高位

        Return
        ===
        `subarray` : 取好的子数组

        Example 
        ===
        ```python
        getarraystr_subarray(['ABC', 'DEF', 'OPQ'], 1)
        ['C', 'F', 'Q']
        ```
        '''
        B = []
        length = len(A)
        for i in range(length):
            B.append(int(str(A[i])[-k]))
        return B

    def countingsort(self, A, k):
        '''
        计数排序，无需比较，非原地排序，时间复杂度`Θ(n)`

        Args
        ===
        `A` : 待排序数组

        `k` : 数组中的元素都不大于k

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> countingsort([0,1,1,3,4,6,5,3,5], 6)
        >>> [0,1,1,3,3,4,5,5,6]
        ```
        '''
        C = []
        B = _deepcopy(A)
        k = 27
        for i in range(k):
            C.append(0)
        length = len(A)
        for j in range(length):
            C[A[j]] = C[A[j]] + 1
        for i in range(1, k):
            C[i] = C[i] + C[i - 1]
        for i in range(length):
            j = length - 1 - i
            B[C[A[j]] - 1] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B

    def radixsort(self, A, d):
        '''
        基数排序 平均时间复杂度为`Θ(nlgn)`

        Args
        ===
        `A` : 待排序的数组

        `d` : 数组A中每个元素都有d位数字/长度,其中第1位是最低位，第d位是最高位

        Return
        ===
        `sortedarray` : 排序好的数组 

        Example
        ===
        ```python
        >>> Chapter8_3().radixsort([54,43,32,21,11], 2)
        >>> [11, 21, 32, 43, 54]
        ```
        '''
        length = len(A)
        B = []
        for i in range(d):
            B.append(self.getarraystr_subarray(A, i + 1))
        for k in range(d):
            B[k] = self.countingsort(B[k], max(B[k]) + 1)
        C = _arange(length)
        for j in range(length):
            for i in range(d):
                C[j] += B[i][j] * 10 ** i
            C[j] = C[j] - j
        return C

    def bucketsort(self, A):
        '''
        桶排序,期望时间复杂度`Θ(n)`(满足输入分布条件`[0,1)`的情况下)
        需要`链表list`额外的数据结构和存储空间

        Args
        ===
        `A` : 待排序的数组

        Return
        ===
        `sortedarray` : 排序好的数组

        Example
        ===
        ```python
        >>> Chapter8_4().bucketsort([0.5, 0.4, 0.3, 0.2, 0.1])
        >>> [0.1, 0.2, 0.3, 0.4, 0.5]
        ```
        '''
        n = len(A)
        B = []
        for i in range(n):
            B.insert(int(n * A[i]), A[i])
        return self.insertsort(B)

    def __find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''
        assert len(kettles1) == len(kettles2)
        n = len(kettles1)
        for i in range(n):
            for j in range(n):
                if kettles1[i] == kettles2[j]:
                    yield (i, j)

    def find_matching_kettle(self, kettles1, kettles2):
        '''
        思考题8.4，找到匹配的水壶，并返回匹配索引集合

        Example
        ===
        ```python
        >>> list(find_matching_kettle([1,2,3,4,5], [5,4,3,2,1]))
        [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        ```
        '''
        return list(self.__find_matching_kettle(kettles1, kettles2))

def quicksort_oneline(arr):
    return arr if len(arr) < 2 else (quicksort_oneline([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + quicksort_oneline([i for i in arr[1:] if i > arr[0]]))

def merge(a, b):
    ret = []
    i = j = 0
    while len(a) >= i + 1 and len(b) >= j + 1:
        if a[i] <= b[j]:
            ret.append(a[i])
            i += 1
        else:
            ret.append(b[j])
            j += 1
    if len(a) > i:
        ret += a[i:]
    if len(b) > j:
        ret += b[j:]
    return ret

def mergesort_easy(arr):
    if len(arr) < 2:
        return arr 
    else: 
        left = mergesort_easy(arr[0 : len(arr) // 2])
        right = mergesort_easy(arr[len(arr) // 2:])
        return merge(left, right)

_inst = Sort()
insertsort = _inst.insertsort
selectsort = _inst.selectsort
bubblesort = _inst.bubblesort
mergesort = _inst.mergesort
heapsort = _inst.heapsort
quicksort = _inst.quicksort
stoogesort = _inst.stoogesort
shellsort = _inst.shellsort

def test():
    '''
    sort.insertsort test

    sort.selectsort test

    sort.bubblesort test

    sort.mergesort test

    sort.heapsort test

    sort.quicksort test
    '''
    print(insertsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(selectsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(bubblesort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(mergesort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(heapsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(quicksort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(shellsort([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(quicksort_oneline([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print(mergesort_easy([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
    print('module sort test successful!!')

if __name__ == '__main__':
    # python src/dugulib/sort.py
    # python3 src/dugulib/sort.py
    test()
else:
    pass


```

```py


from __future__ import absolute_import, print_function

from copy import deepcopy as _deepcopy

import time as _time
from random import randint as _randint

class SearchTreeNode:
    '''
    二叉查找树的结点
    '''
    def __init__(self, key, index, \
        p = None, left = None, right = None):
        '''

        二叉树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.p = p

    def __str__(self):
        return 'key:' + str(self.key) + ','\
                'index:' + str(self.index)

class SearchTree:
    '''
    二叉查找树
    '''
    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []

    def inorder_tree_walk(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None:
            array.append(str(x))
            array = array + right
        return array

    def __inorder_tree_walk_key(self, x : SearchTreeNode):
        '''
        从二叉查找树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.__inorder_tree_walk_key(x.left)
            array = array + left
            right = self.__inorder_tree_walk_key(x.right)  
        if x != None:
            array.append(x.key)
            array = array + right
        return array

    def tree_search(self, x : SearchTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != None and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return None

    def iterative_tree_search(self, x : SearchTreeNode, key):
        '''
        查找的非递归版本

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x != None:
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x
        return x

    def minimum(self, x : SearchTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != None:
            x = x.left
        return x

    def __minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : SearchTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def maximum(self, x : SearchTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != None:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != None:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : SearchTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return None

    def successor(self, x : SearchTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.right != None:
            return self.minimum(x.right)
        y = x.p
        while y != None and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : SearchTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != None:
            return self.maximum(x.left)
        y = x.p
        while y != None and x == y.left:
            x = y
            y = y.p
        return y

    def insertkey(self, key, index = None):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        self.insert(SearchTreeNode(key, index))

    def insert(self, z : SearchTreeNode):
        '''
        插入元素，时间复杂度`O(h)` `h`为树的高度
        '''
        y = None
        x = self.root
        while x != None:
            y = x
            if z.key < x.key:
                x = x.left
            elif z.key > x.key:
                x = x.right
            else:
                # 处理相同结点的方式，随机分配左右结点
                if _randint(0, 1) == 0:
                    x = x.left
                else:
                    x = x.right
        z.p = y
        if y == None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        elif z.key > y.key:
            y.right = z
        else:
            # 处理相同结点的方式，随机分配左右结点
            if _randint(0, 1) == 0:
                y.left = z
            else:
                y.right = z
        self.nodes.append(z) 
        self.lastnode = z

    def insertnodes(self, nodes : list):
        '''
        按顺序插入一堆结点
        '''
        for node in nodes:
            if node is type(SearchTreeNode):
                self.insert(node)
            else:
                self.insertkey(node)

    def __insertfrom(self, z : SearchTreeNode, x : SearchTreeNode, lastparent : SearchTreeNode):
        if x != None:
            if z.key < x.key:
                self.__insertfrom(z, x.left, x)
            else:
                self.__insertfrom(z, x.right, x)
        else:
            z.p = lastparent
            if z.key < lastparent.key:
                lastparent.left = z
            else:
                lastparent.right = z

    def insert_recursive(self, z : SearchTreeNode):
        '''
        插入元素(递归版本)，时间复杂度`O(h)` `h`为树的高度
        '''
        if self.root == None:
            self.root = z
        else:  
            self.__insertfrom(z, self.root, None)
        self.nodes.append(z) 
        self.lastnode = z

    def delete(self, z : SearchTreeNode):
        '''
        删除操作，时间复杂度`O(h)` `h`为树的高度
        '''
        if z.left == None or z.right == None:
            y = z
        else:
            y = self.successor(z)
        if y.left != None:
            x = y.left
        else:
            x = y.right
        if x != None:
            x.p = y.p
        if y.p == None:
            self.root = x
        else:
            if y == y.p.left:
                y.p.left = x
            else:
                y.p.right = x
        if y != None:
            z.key = y.key
            z.index = _deepcopy(y.index)
        self.nodes.remove(z) 
        return y
        
    def all(self):
        '''
        返回二叉查找树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({ "index":node.index,"key" : node.key})
        return array

    def allkey(self):
        '''
        按升序的方式输出所有结点`key`值构成的集合
        '''
        return self.__inorder_tree_walk_key(self.root)

    def count(self):
        '''
        二叉查找树中的结点总数
        '''
        return len(self.nodes)

    def leftrotate(self, x : SearchTreeNode):
        '''
        左旋 时间复杂度:`O(1)`
        '''
        if x.right == None:
            return
        y : SearchTreeNode = x.right
        x.right = y.left
        if y.left != None:
            y.left.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y

    def rightrotate(self, x : SearchTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        if x.left == None:
            return
        y : SearchTreeNode = x.left
        x.left = y.right
        if y.right != None:
            y.right.p = x
        y.p = x.p
        if x.p == None:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y

class RandomSearchTree(SearchTree):

    def __init__(self):
        self.lastnode : SearchTreeNode = None
        self.root : SearchTreeNode = None
        self.nodes = []
        self.__buffers = []

    def __randomize_inplace(self, array):
        '''
        随机打乱排列一个数组

        Args
        ===
        `array` : 随机排列前的数组

        Return
        ===
        `random_array` : 随机排列后的数组

        '''
        n = len(array)
        for i in range(n):
            rand = _randint(i, n - 1)
            _time.sleep(0.001)
            array[i], array[rand] = array[rand], array[i]
        return array

    def randominsert(self, z : SearchTreeNode):
        '''
        使用随机化技术插入结点到缓存
        '''
        self.__buffers.append(z)

    def randominsertkey(self, key, index = None):
        '''
        使用随机化技术插入结点到缓存
        '''
        z = SearchTreeNode(key, index)
        self.randominsert(z)

    def update(self):
        '''
        从缓存更新二叉查找树结点
        '''
        randombuffers = self.__randomize_inplace(self.__buffers)
        for buffer in randombuffers:
            self.insert(buffer)
        self.__buffers.clear()

BLACK = 0
RED = 1

class RedBlackTreeNode:
    '''
    红黑树结点
    '''
    def __init__(self, key, index = None, color = RED, \
        p = None, left = None, right = None):
        '''
        红黑树树结点

        Args
        ===
        `left` : SearchTreeNode : 左儿子结点

        `right`  : SearchTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `p` : 父节点

        '''
        self.left = left
        self.right = right
        self.key = key
        self.index = index
        self.color = color
        self.p = p

    def __str__(self):
        '''
        str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})
        '''
        return  str({'key' : self.key, 
            'index' : self.index, 
            'color' : self.color})

    def isnil(self):
        '''
        判断红黑树结点是否是哨兵结点
        '''
        if self.key == None and self.color == BLACK:
            return True
        return False

class RedBlackTree:
    '''
    红黑树
    '''
    def __init__(self):
        '''
        红黑树
        '''
        self.nil = self.buildnil()
        self.root = self.nil

    def buildnil(self):
        '''
        构造一个新的哨兵nil结点
        '''
        nil = RedBlackTreeNode(None, color=BLACK)
        return nil

    def insertkey(self, key, index = None, color = RED):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        z = RedBlackTreeNode(key, index, color)
        self.insert(z)

    def successor(self, x : RedBlackTreeNode):
        '''
        前趋:结点x的前趋即具有小于x.key的关键字中最大的那个

        时间复杂度：`O(h)`, `h=lgn`为树的高度
        
        '''
        if x.right != self.nil:
            return self.minimum(x.right)
        y = x.p
        while y != self.nil and x == y.right:
            x = y
            y = y.p
        return y

    def predecessor(self, x : RedBlackTreeNode):
        '''
        后继:结点x的后继即具有大于x.key的关键字中最小的那个

        时间复杂度：`O(h)`, `h`为树的高度
        
        '''
        if x.left != self.nil:
            return self.maximum(x.left)
        y = x.p
        while y != self.nil and x == y.left:
            x = y
            y = y.p
        return y

    def tree_search(self, x : RedBlackTreeNode, key):
        '''
        查找 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        try:
            if x != self.nil and key == x.key:
                return x
            if key < x.key:
                return self.tree_search(x.left, key)
            else:
                return self.tree_search(x.right, key)            
        except :
            return self.nil

    def minimum(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(迭代版本) 

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.left != self.nil:
            x = x.left
        return x

    def __minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__minimum_recursive(x.left)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def minimum_recursive(self, x : RedBlackTreeNode):
        '''
        最小关键字元素(递归版本) 

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__minimum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def maximum(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(迭代版本)

        时间复杂度：`O(h)`, `h`为树的高度

        '''
        while x.right != self.nil:
            x = x.right
        return x
    
    def __maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = []
        if x != self.nil:
            ex = self.__maximum_recursive(x.right)
            if ex == []:
                z = x
                array.append(z)
                return array
            else:
                array = array + ex
        return array

    def maximum_recursive(self, x : RedBlackTreeNode):
        '''
        最大关键字元素(递归版本)

        时间复杂度：`O(h)`, `h`为树的高度
        '''
        array = self.__maximum_recursive(x)
        if len(array) != 0:
            return array.pop()
        return self.nil

    def insert(self, z : RedBlackTreeNode):
        '''
        插入红黑树结点 时间复杂度 `O(lgn)`
        '''
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = RED
        self.insert_fixup(z)

    def insert_fixup(self, z : RedBlackTreeNode):
        '''
        插入元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while z.p.color == RED:
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == RED:
                    z.p.color = BLACK
                    y.color = BLACK
                    z.p.p.color = RED
                    z = z.p.p
                elif y.color == BLACK and z == z.p.right:
                    z = z.p
                    self.leftrotate(z)
                elif y.color == BLACK and z == z.p.left:
                    z.p.color = BLACK
                    z.p.p.color = RED
                    self.rightrotate(z.p.p)               
        self.root.color = BLACK    
        
    def delete_fixup(self, x : RedBlackTreeNode):
        '''
        删除元素后 修正红黑树性质，结点重新旋转和着色
        '''
        while x != self.root and x.color == BLACK:
            if x == x.p.left:
                w : RedBlackTreeNode = x.p.right
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.leftrotate(x.p)
                    w = x.p.right
                elif w.color == BLACK:
                    if w.left.color == BLACK and w.right.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self.rightrotate(w)
                        w = x.p.right
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.right.color = BLACK
                        self.leftrotate(x.p)
                        x = self.root
            else:
                w : RedBlackTreeNode = x.p.left
                if w.color == RED:
                    w.color = BLACK
                    x.p.color = RED
                    self.rightrotate(x.p)
                    w = x.p.left
                elif w.color == BLACK:
                    if w.right.color == BLACK and w.left.color == BLACK:
                        w.color = RED
                        x = x.p
                    elif w.left.color == RED and w.right.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.leftrotate(w)
                        w = x.p.left
                    elif w.right.color == RED:
                        w.color = x.p.color
                        x.p.color = BLACK
                        w.left.color = BLACK
                        self.rightrotate(x.p)
                        x = self.root
        x.color = BLACK

    def delete(self, z : RedBlackTreeNode):
        '''
        删除红黑树结点
        '''
        if z.isnil() == True:
            return
        if z.left == self.nil or z.right == self.nil:
            y = z
        else:
            y = self.successor(z)
        if y.left != self.nil:
            x = y.left
        else:
            x = y.right
        x.p = y.p
        if x.p == self.nil:
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        if y != z:
            z.key = y.key
            z.index = _deepcopy(y.index)
        if y.color == BLACK:
            self.delete_fixup(x)
        return y
    
    def deletekey(self, key):
        '''
        删除红黑树结点
        '''
        node = self.tree_search(self.root, key)
        return self.delete(node)

    def leftrotate(self, x : RedBlackTreeNode):
        '''
        左旋 时间复杂度: `O(1)`
        '''
        y : RedBlackTreeNode = x.right
        z = y.left
        if y == self.nil:
            return 
        y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        x.right = z

    def rightrotate(self, x : RedBlackTreeNode):
        '''
        右旋 时间复杂度:`O(1)`
        '''
        y : RedBlackTreeNode = x.left
        z = y.right
        if y == self.nil:
            return
        y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y
        x.left = z
            
    def inorder_tree_walk(self, x : RedBlackTreeNode):
        '''
        从红黑树的`x`结点后序遍历
        '''
        array = []
        if x != None:
            left = self.inorder_tree_walk(x.left)
            array = array + left
            right = self.inorder_tree_walk(x.right)  
        if x != None and x.isnil() == False:
            array.append(str(x))
            array = array + right
        return array
    
    def all(self):
        '''
        按`升序` 返回红黑树中所有的结点
        '''
        return self.inorder_tree_walk(self.root)

    def clear(self):
        '''
        清空红黑树
        '''
        self.destroy(self.root)
        self.root = self.buildnil()

    def destroy(self, x : RedBlackTreeNode):
        '''
        销毁红黑树结点
        '''
        if x == None:
            return
        if x.left != None:   
            self.destroy(x.left)
        if x.right != None:  
            self.destroy(x.right) 
        x = None
  
    def __preorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            print(str(node), end=' ')  
            self.__preorder(node.left) 
            self.__preorder(node.right)  

    def __inorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left) 
            print(str(node), end=' ') 
            self.__preorder(node.right)  

    def __postorder(self, node : RedBlackTreeNode): 
        if node.isnil() == False:
            self.__preorder(node.left)       
            self.__preorder(node.right) 
            print(str(node), end=' ') 

    def preorder_print(self):
        '''
        前序遍历红黑树
        ''' 
        print('preorder')
        self.__preorder(self.root)
        print('')

    def inorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('inorder')
        self.__inorder(self.root)
        print('')

    def postorder_print(self):
        '''
        中序遍历红黑树
        '''
        print('postorder')
        self.__postorder(self.root)
        print('')

    @staticmethod
    def test():
        tree = RedBlackTree()
        tree.insertkey(41)
        tree.insertkey(38)
        tree.insertkey(31)
        tree.insertkey(12)
        tree.insertkey(19)
        tree.insertkey(8)
        tree.insertkey(1)
        tree.deletekey(12)
        tree.deletekey(38)
        tree.preorder_print()
        tree.postorder_print()
        tree.inorder_print()
        print(tree.all())
        tree.clear()
        print(tree.all())


if __name__ == '__main__':
    tree = SearchTree()
    node1 = SearchTreeNode(12, 0)
    node2 = SearchTreeNode(11, 1)
    node3 = SearchTreeNode(10, 2)
    node4 = SearchTreeNode(15, 3)
    node5 = SearchTreeNode(9, 4)
    tree.insert_recursive(node1)
    tree.insert(node2)
    tree.insert(node3)
    tree.insert(node4)
    tree.insert_recursive(node5)   
    print(tree.all())
    print(tree.count())
    print(tree.inorder_tree_walk(tree.root))
    print(tree.tree_search(tree.root, 15))
    print(tree.tree_search(tree.root, 8))
    print(tree.iterative_tree_search(tree.root, 10))
    print(tree.iterative_tree_search(tree.root, 7))
    print(tree.maximum(tree.root))
    print(tree.maximum_recursive(tree.root))
    print(tree.minimum(tree.root))
    print(tree.minimum_recursive(tree.root))
    print(tree.successor(tree.root))
    print(tree.predecessor(tree.root))
    tree.insertkey(18)
    tree.insertkey(16)
    tree.leftrotate(node4)
    tree.insertkey(20)
    tree.rightrotate(node3)
    tree.insertkey(3)
    print(tree.all())
    random_tree = RandomSearchTree()
    random_tree.randominsertkey(1)
    random_tree.randominsertkey(2)
    random_tree.randominsertkey(3)
    random_tree.randominsertkey(4)
    random_tree.randominsertkey(5)
    random_tree.update()
    random_tree.insertkey(0)
    print(random_tree.all())
    print(random_tree.allkey())
    print(random_tree.inorder_tree_walk(random_tree.root))

    RedBlackTree.test()

    # python src/dugulib/tree.py
    # python3 src/dugulib/tree.py
else:
    pass


```
