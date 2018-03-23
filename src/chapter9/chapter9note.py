
# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
'''
Class Chapter9_1

Class Chapter9_2

Class Chapter9_3

Class Chapter9_4
'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter9_1:
    '''
    chpater9.1 note and function
    '''

    def minimum(self, A : list) -> float:
        '''
        求集合中的最小值
        '''
        min = A[0]
        for i in range(1, len(A)):
            if min > A[i]:
                min = A[i]
        return min

    def note(self):
        '''
        Summary
        ====
        Print chapter9.1 note

        Example
        ====
        ```python
        Chapter9_1().note()
        ```
        '''
        print('chapter9.1 note as follow')
        print('第9章 中位数和顺序统计学')
        print('在一个由n个元素组成的集合中，第i个顺序统计量是该集合中第i小的元素。')
        print('非形式地说：一个中位数是它所在集合的\"中点元素\"')
        print('当n为奇数时，中位数是唯一的，出现在i=(n+1)/2处；')
        print('当n为偶数时，存在两个中位数，分别出现在i=n/2和i=n/2+1处')
        print('不考虑n的奇偶性，中位数总是出现在i=[(n+1)/2]处(下中位数)和i=[(n+1)/2]处(上中位数)')
        print('简单起见，书中的中位数总是指下中位数')
        print('本章讨论从一个由n个不同数值构成的集合中选择其第i个顺序统计量的问题，假设集合中的数互异')
        print('如下形式化地定义选择问题：')
        print(' 输入：一个包含n个(不同的)数的集合A和一个数i,1<=i<=n')
        print(' 输出：元素x属于A，它恰好大于A中其他i-1个数')
        print('选择问题可以在O(nlgn)时间内解决，因为可以用堆排序或合并排序对输入数据进行排序')
        print(' 然后在输出数组中标出第i个元素即可。但是还有其他更快的方法')
        print('9.1 最小值和最大值')
        A = [61, 52, 43, 34, 25, 16, 17]
        print('数组A', _deepcopy(A), '中元素的最小的元素为:', self.minimum(A))
        print('可以通过n-1次比较找出一个数组中的上界和下界')
        print('在某些应用中同时找出最小值和最大值')
        print('要设计出一个算法，使之通过渐进最优的Θ(n)次比较，能从n个元素中找出最小值和最大值')
        print('只要独立地找出最小值和最大值，各用n-1次比较，共有2n-2次比较')
        print('事实上，至多3[n/2]次比较久足以同时找出最小值和最大值，做法是记录比较过程中遇到的最大值和最小值')
        print('练习9.1-1 在最坏情况下，利用n+[lgn]-2次比较，即可找到n个元素中的第2小元素')
        print('练习9.1-2 在最坏情况下，同时找到n个数字中的最大值和最小值需要[3n/2]-2次比较')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_2:
    '''
    chpater9.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter9.2 note

        Example
        ====
        ```python
        Chapter9_2().note()
        ```
        '''
        print('chapter9.2 note as follow')
        print('')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_3:
    '''
    chpater9.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter9.3 note

        Example
        ====
        ```python
        Chapter9_3().note()
        ```
        '''
        print('chapter9.3 note as follow')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py

class Chapter9_4:
    '''
    chpater9.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter9.4 note

        Example
        ====
        ```python
        Chapter9_4().note()
        ```
        '''
        print('chapter9.4 note as follow')
        # python src/chapter9/chapter9note.py
        # python3 src/chapter9/chapter9note.py


chapter9_1 = Chapter9_1()
chapter9_2 = Chapter9_2()
chapter9_3 = Chapter9_3()
chapter9_4 = Chapter9_4()

def printchapter9note():
    '''
    print chapter9 note.
    '''
    print('Run main : single chapter nine!')  
    chapter9_1.note()
    chapter9_2.note()
    chapter9_3.note()
    chapter9_4.note()

# python src/chapter9/chapter9note.py
# python3 src/chapter9/chapter9note.py
if __name__ == '__main__':  
    printchapter9note()
else:
    pass
