
# python src/chapter6/chapter6_3.py
# python3 src/chapter6/chapter6_3.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

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
        # python src/chapter6/chapter6_2.py
        # python3 src/chapter6/chapter6_2.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_3().note()
else:
    pass
