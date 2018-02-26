
# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py

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
        print('代换法解递归式的两个步骤')
        # python src/chapter4/chapter4_1.py
        # python3 src/chapter4/chapter4_1.py
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_1().note()
else:
    pass
