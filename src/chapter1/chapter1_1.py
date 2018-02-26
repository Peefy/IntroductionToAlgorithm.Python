
## python src/chapter_1/chapter1_1.py
## python3 src/chapter_1/chapter1_1.py

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

