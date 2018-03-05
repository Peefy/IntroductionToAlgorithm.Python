
# python src/chapter6/chapter6_1.py
# python3 src/chapter6/chapter6_1.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter6_1:
    '''
    CLRS 第六章 6.1 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter6.1 note

        Example
        =
        >>> Chapter6_1().note()
        '''
        print('第二部分 排序和顺序统计学')
        print('这一部分将给出几个排序问题的算法')
        print('排序算法是算法学习中最基本的问题')
        print('排序可以证明其非平凡下界的问题')
        print('最佳上界可以与这个非平凡下界面渐进地相等，意味者排序算法是渐进最优的')
        print('在第二章中，插入排序的复杂度虽然为Θ(n^2)，但是其内循环是最为紧密的，对于小规模输入可以实现快速的原地排序')
        print('并归排序的复杂度为Θ(nlgn)，但是其中的合并操作不在原地进行')
        print('第六章介绍堆排序，第七章介绍快速排序')
        print('堆排序用到了堆这个数据结构，还要用它实现优先级队列')
        print('插入排序，合并排序，堆排序，快速排序都是比较排序')
        print('n个输入的比较排序的下界就是Ω(nlgn)，堆排序和合并排序都是渐进最优的比较排序')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter6/chapter6_1.py
        # python3 src/chapter6/chapter6_1.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_1().note()
else:
    pass
