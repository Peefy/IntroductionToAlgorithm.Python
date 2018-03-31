
# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
'''
Class Chapter11_1

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
        print('但在实践中，散列技术的效率是很高的。在一些合理的假设下，在散列表中查找一个元素的期望时间为O(1)')
        print('散列表是普通数组概念的推广，因为可以对数组进行直接寻址，故可以在O(1)时间内访问数组的任意元素')
        print('如果存储空间允许，可以提供一个数组，为每个可能的关键字保留一个位置，就可以应用直接寻址技术')
        print('当实际存储的关键字数比可能的关键字总数较小时，这是采用散列表就会较直接数组寻址更为有效')
        print('在散列表中，不是直接把关键字用作数组下标，而是根据关键字计算出下标。')
        print('11.2着重介绍解决碰撞的链接技术。')
        print('所谓碰撞，就是指多个关键字映射到同一个数组下标位置')
        print('11.3介绍如何利用散列函数，根据关键字计算出数组的下标。')
        print('11.4介绍开放寻址法，它是处理碰撞的另一种方法。散列是一种极其有效和实用的技术，基本的字典操作只需要O(1)的平均时间')
        print('11.5解释当待排序的关键字集合是静态的，\"完全散列\"如何能够在O(1)最坏情况时间内支持关键字查找')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_2:
    '''
    chpater11.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.2 note

        Example
        ====
        ```python
        Chapter11_2().note()
        ```
        '''
        print('chapter11.2 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_3:
    '''
    chpater11.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.3 note

        Example
        ====
        ```python
        Chapter11_3().note()
        ```
        '''
        print('chapter11.3 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_4:
    '''
    chpater11.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.4 note

        Example
        ====
        ```python
        Chapter11_4().note()
        ```
        '''
        print('chapter11.4 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_5:
    '''
    chpater11.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.5 note

        Example
        ====
        ```python
        Chapter11_5().note()
        ```
        '''
        print('chapter11.5 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

chapter11_1 = Chapter11_1()
chapter11_2 = Chapter11_2()
chapter11_3 = Chapter11_3()
chapter11_4 = Chapter11_4()
chapter11_5 = Chapter11_5()

def printchapter11note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter eleven!')  
    chapter11_1.note()
    chapter11_2.note()
    chapter11_3.note()
    chapter11_4.note()
    chapter11_5.note()

# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
if __name__ == '__main__':  
    printchapter11note()
else:
    pass
