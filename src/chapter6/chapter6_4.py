
# python src/chapter6/chapter6_4.py
# python3 src/chapter6/chapter6_4.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import heap
else:
    from . import heap
    
class Chapter6_4:
    '''
    CLRS 第六章 6.4 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_4().note()
        '''
        print('6.4 堆排序算法')
        # python src/chapter6/chapter6_4.py
        # python3 src/chapter6/chapter6_4.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_4().note()
else:
    pass
