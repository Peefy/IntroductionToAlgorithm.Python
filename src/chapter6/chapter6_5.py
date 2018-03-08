
# python src/chapter6/chapter6_5.py
# python3 src/chapter6/chapter6_5.py

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
    
class Chapter6_5:
    '''
    CLRS 第六章 6.5 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.4 note

        Example
        =
        >>> Chapter6_5().note()
        '''
        print('6.5 优先级队列')

        # python src/chapter6/chapter6_5.py
        # python3 src/chapter6/chapter6_5.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_5().note()
else:
    pass
