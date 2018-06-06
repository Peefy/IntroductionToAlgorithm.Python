# coding:utf-8
# usr/bin/python3
# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
'''

Class Chapter17_1

Class Chapter17_2

Class Chapter17_3

Class Chpater17_4

Class Chapter17_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array
from numpy import * 

class Chapter17_1:
    '''
    chpater17.1 note and function
    '''
    def multipop(self, S : list, k):
        '''
        栈的弹出多个数据的操作
        '''
        while len(S) > 0 and k > 0:
            S.pop()
            k = k - 1

    def increment(self, A : list):
        i = 0
        while i < len(A) and A[i] == 1:
            A[i] = 0
            i += 1
        if i < len(A):
            A[i] = 1

    def note(self):
        '''
        Summary
        ====
        Print chapter17.1 note

        Example
        ====
        ```python
        Chapter17_1().note()
        ```
        '''
        print('chapter17.1 note as follow')  
        # !在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的
        print('在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的')
        # !平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能
        print('平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能')
        # !平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法
        print('平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法')
        print('17.1 聚集分析')
        print('在聚集分析中，要证明对所有的n,由n个操作所构成的序列的总时间在最坏的情况下为T(n).')
        print('因此，在最坏情况下，每个操作的平均代价(或称平摊代价)为T(n)/n')
        print('这个平摊代价对每个操作都是成立的，即使当序列中存在几种类型的操作也是一样的')
        print('例1.栈操作')
        print(' 10.1介绍了两种基本的栈操作，每种操作的时间代价都是O(1)：PUSH和POP操作')
        print(' 因此，含n个PUSH和POP操作的序列的总代价为n,而这n个操作的实际运行时间就是Θ(n)')
        print(' 现在增加一个栈操作MULTIPOP(S,k),它的作用使弹出栈S的k个栈顶对象，或者当栈包含少于k个对象时，弹出整个栈中的数据对象')
        print(' 分析一个由n个PUSH，POP和MULTIPOP操作构成的序列，其作用于一个初始为空的栈。')
        print(' 序列中一次MULTIPOP操作的最坏情况代价为O(n),因为栈的大小至多为n')
        print(' 因此，任意栈操作的最坏情况就是O(n),因此n个操作的序列的代价是O(n^2),因为可能会有O(n)个MULTIPOP操作，每个的代价都是O(n)')
        print(' 利用聚集分析，可以获得一个考虑到n个操作的整个序列的更好的上界')
        print(' 对任意的n值，包含n个PUSH,POP和MULTIPOP操作的序列的总时间为O(n).每个操作的平均代价为O(n)/n=O(1)')
        print(' 把每个操作的平摊代价指派为平均代价。在这个例子中，三个栈操作的平摊代价都是O(1)')
        print('例2.二进制计数器递增1')
        print(' 作为聚集分析的另一个例子，考虑实现一个由0开始向上计数的k位二进制计数器的问题')
        print(' 使用一个位数组A[0..k-1]作为计数器。存储在计数器中的一个二进制数x的最低位在A[0]中，最高位在A[k-1]')
        print(' 每次INCREMENT操作的代价都与被改变值的位数成线性关系')
        print(' 如同栈的例子，大致的分析只能得到正确但不紧确的界')
        print(' 在最坏的情况下，INCREMENT的每次执行要花Θ(nk)')
        print(' 注意到在每次调用INCREMENT时，并不是所有的位都翻转，可以分析得更紧确一些')
        print(' 来得到n次INCREMENT操作的序列的最坏情况代价为O(n)')
        print(' 在每次调用INCREMENT时，A[0]确实都要发生翻转，下一个高位A[1]每隔一次翻转',
            '当作用于初始为零的计数器上时，n次INCREMENT操作会导致A[1]翻转[n/2]次')
        print(' 对于i>[lgn],位A[i]始终保持不变。在序列中发生的位翻转的总次数为2n')
        print(' 所以在最坏的情况下，作用于一个初始为零的计数器上的n次INCREMENTC操作的时间为O(n)。')
        print(' 每次操作的平均代价(即每次操作的平摊代价)是O(n)/n=O(1)')
        print(' 练习17.1-1 如果一组栈操作中包括了一次MULTIPUSH操作，它一次把k个元素压入栈内，',
            '那么栈操作的平摊代价的界O(1)还能够保持')
        print(' 练习17.1-2 证明：在k位计数器的例子中，如果包含一个DECREMENT操作，n个操作可能花费Θ(nk)的时间')
        print(' 练习17.1-3 最坏情况是这个数据结构的每个数据都是2的整数幂，运行时间为O(n),聚集分析的每次操作的平摊代价为O(n)/n=1')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_2:
    '''
    chpater17.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.2 note

        Example
        ====
        ```python
        Chapter17_2().note()
        ```
        '''
        print('chapter17.2 note as follow')
        print('17.2 记账方法')
        print('在平摊分析的记账方法中。对不同的操作赋予不同的费用，某些操作的费用比它们的实际代价或多或少')
        print('对一个操作的收费的数量称为平摊代价')
        print('当一个操作的平摊代价超过了它的实际代价时，两者的差值就被当做存款(credit),并赋予数据结构中的一些特定对象')
        print('存款可以在以后用于补偿那些平摊代价低于其实际代价的操作')
        # !就可以将一个操作的平摊代价看做两部分：其实际代价与存款
        print('就可以将一个操作的平摊代价看做两部分：其实际代价与存款')
        print('记账方法与聚集方法有着很大的不同，对后者而言，所有操作都具有相同的平摊代价')
        print('选择操作的平摊代价必须很小心。如果希望通过对平摊代价的分析来说明每次操作的最坏情况平均代价较小,',
            '则操作序列的总的平摊代价就必须是该序列的总的实际代价的一个上界')
        print('记账方法和聚集方法一样，这种关系必须对所有的操作序列都成立')
        print('例1.栈操作')
        print(' 对于栈操作例子的平摊分析的记账方法，各个栈操作的实际代价为')
        print('  PUSH 1;')
        print('  POP 1;')
        print('  MULTIPOP min(k, s)')
        print(' 其中k为MULTIPOP的一个参数,s为调用该操作时栈的大小。现在对它们赋值以下的平摊代价')
        print('  PUSH 2;')
        print('  POP 0;')
        print('  MULTIPOP 0;')
        print(' 注意MULTIPOP的平摊代价是常数(0),而它的实际代价却是个变量,此处所有三个平摊代价都是O(1)')
        print(' 但一般来说，从渐进的意义上看，所考虑的各种操作的平摊代价是会发生变化的')
        print('结论：对任意的包含n次PUSH，POP和MULTIPOP操作的序列，总的平摊代价就是其总的实际代价的一个上界')
        print(' 又因为总的平摊代价为O(n),故总的实际代价为O(n)')
        print('例2.二进制计数器递增1')
        # !二进制计数器递增1这个操作的运行时间与发生翻转的位数是成正比的
        print(' 位数在本例中将被用作代价。用1元钱表示单位代价(即某一位的翻转)')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_3:
    '''
    chpater17.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.3 note

        Example
        ====
        ```python
        Chapter17_3().note()
        ```
        '''
        print('chapter17.3 note as follow')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_4:
    '''
    chpater17.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.4 note

        Example
        ====
        ```python
        Chapter17_4().note()
        ```
        '''
        print('chapter17.4 note as follow')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

chapter17_1 = Chapter17_1()
chapter17_2 = Chapter17_2()
chapter17_3 = Chapter17_3()
chapter17_4 = Chapter17_4()

def printchapter17note():
    '''
    print chapter17 note.
    '''
    print('Run main : single chapter seventeen!')  
    chapter17_1.note()
    chapter17_2.note()
    chapter17_3.note()
    chapter17_4.note()

# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
if __name__ == '__main__':  
    printchapter17note()
else:
    pass
