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
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
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
