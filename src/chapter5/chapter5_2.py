
# python src/chapter5/chapter5_1.py
# python3 src/chapter5/chapter5_1.py

import sys as _sys
import math as _math

import random as _random

from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_2:
    '''
    CLRS 第五章 5.2 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter5.2 note

        Example
        =
        >>> Chapter5_2().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.2 指示器随机变量')

        # python src/chapter5/chapter5_2.py
        # python3 src/chapter5/chapter5_2.py
        return self

_instance = Chapter5_2()
note = _instance.note  

if __name__ == '__main__':
    print('Run main : single chapter five!')
    Chapter5_2().note()
else:
    pass
