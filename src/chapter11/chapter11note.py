
# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
'''
Class Chapter10_1

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter11_1:
    '''
    chpater11.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.1 note

        Example
        ====
        ```python
        Chapter11_1().note()
        ```
        '''
        print('chapter11.1 note as follow')
        print('第11章 散列表')
        print('在很多应用中，都要用到一种动态集合结构，它仅支持INSERT,SEARCH的DELETE字典操作')
        print('实现字典的一种有效数据结构为散列表(HashTable)')
        print('在最坏情况下，在散列表中，查找一个元素的时间在与链表中查找一个元素的时间相同')
        print('在最坏情况下都是Θ(n)')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

chapter11_1 = Chapter11_1()

def printchapter11note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter eleven!')  
    chapter11_1.note()

# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
if __name__ == '__main__':  
    printchapter11note()
else:
    pass
