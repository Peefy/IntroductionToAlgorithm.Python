
# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
'''
Class Chapter8_1

Class Chapter8_2

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter8_1:
    def note(self):
        '''
        Summary
        ====
        Print chapter8.1 note

        Example
        ====
        >>> Chapter8_1().note()
        '''
        print('chapter8.1 note as follow')
        print('第8章 线性时间排序')
        print('合并排序和堆排序在最坏情况下能达到O(nlgn),快速排序在平均情况下达到此上界面')
        print('第8章之前的排序算法都是比较排序')
        print('8.1节中将证明对含有n个元素的一个输入序列，', 
            '任何比较排序在最坏情况下都要用Ω(nlgn)次比较来进行排序')
        print('')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

chapter8_1 = Chapter8_1()

def printchapter8note():
    '''
    print chapter8 note.
    '''
    print('Run main : single chapter eight!')  
    chapter8_1.note()

# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
if __name__ == '__main__':  
    printchapter8note()
else:
    pass
