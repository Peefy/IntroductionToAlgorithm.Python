
# python src/chapter3/chapter3_2.py
# python3 src/chapter3/chapter3_2.py

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
    CLRS 第三章 3.1 算法函数和笔记
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
        print('多重对数函数：用记号lg * n(读作n的log星)来表示多重对数，定义为lg * n=min(i>=0,)')
        # python src/chapter3/chapter3_2.py
        # python3 src/chapter3/chapter3_2.py
        
if __name__ == '__main__':
    print('Run main : single chapter three!')
    Chapter3_2().note()
else:
    pass
