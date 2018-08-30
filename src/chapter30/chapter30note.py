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
