
# python src/chapter6/chapter6_1.py
# python3 src/chapter6/chapter6_1.py

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
        # python src/chapter6/chapter6_1.py
        # python3 src/chapter6/chapter6_1.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_1().note()
else:
    pass
