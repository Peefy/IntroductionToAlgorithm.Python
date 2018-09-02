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
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
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
