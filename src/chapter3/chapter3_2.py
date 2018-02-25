
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
        print('多项式定义')
        print('指数式定义')
        # python src/chapter3/chapter3_2.py
        # python3 src/chapter3/chapter3_2.py
        
if __name__ == '__main__':
    print('Run main : single chapter three!')
    Chapter3_2().note()
else:
    pass
