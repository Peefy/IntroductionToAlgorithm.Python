
# python src/chapter5/chapter5_4.py
# python3 src/chapter5/chapter5_4.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_4:
    '''
    CLRS 第五章 5.4 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter5.4 note

        Example
        =
        >>> Chapter5_4().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.4 概率分析和指示器随机变量的进一步使用')

        # python src/chapter5/chapter5_4.py
        # python3 src/chapter5/chapter5_4.py
        return self

_instance = Chapter5_4()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_4().note()
else:
    pass
