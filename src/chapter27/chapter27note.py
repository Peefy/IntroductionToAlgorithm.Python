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
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
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
