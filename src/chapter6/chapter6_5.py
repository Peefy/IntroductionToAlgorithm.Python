
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
        print('虽然堆排序算法是一个很漂亮的算法，快速排序(第7章将要介绍)的一个好的实现往往优于堆排序')
        print('堆数据结构还是有着很大的用处，一个很常见的应用：作为高效的优先级队列')
        print('和堆一样，队列也有两种，最大优先级队列和最小优先级队列')
        print('优先级队列是一种用来维护由一组元素构成的集合S的数据结构，这一组元素中的每一个都有一个关键字key')
        print('一个最大优先级队列支持一下操作')
        print('INSERT(S, x):把元素x插入集合S，这一操作可写为S<-S∪{x}')
        print('MAXIMUM(S)：返回S中具有最大关键字')
        print('EXTRACT-MAX(S)：去掉并返回S中的具有最大关键字的元素')
        print('INCREASE-KEY(S，x, k):将元素x的关键字的值增加到k，这里k值不能小于x的原关键字的值')
        print('最大优先级队列的一个应用是在一台分时计算机上进行作业调度。')
        print('当一个作业做完或被中断时，用EXTRACT-MAX操作从所有等待的作业中，选择出具有最高优先级的作业')
        print('在任何时候，一个新作业都可以用INSERT加入到队列中去')
        print('当用堆来实现优先级队列时，需要在堆中的每个元素里存储对应的应用对象的柄handle，',
            '对象柄的准确表示到底怎样(一个指针或者一个整形数)还取决于具体的应用')
        A = [14, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-1 数组A=', _deepcopy(A), '执行HEAP-EXTRACT-MAX操作的过程为：', heap.extractmax(A), A)
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print(' 数组A=', _deepcopy(A), '执行HEAP-INCREASE-KEY操作的过程为：', heap.increasekey(A, 2, 16))
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-2 数组A=', _deepcopy(A), '执行MAX-HEAP-INSERT(A, 10)操作的过程为：', heap.maxheapinsert(A, 10))
        print('练习6.5-3 基本把最大堆算法的不等号方向改以以下就可以')
        print('练习6.5-4 因为插入的元素并不知道其大小和插入前原始最大堆元素的大小比较情况，所以将插入的数据放到二叉堆的最底部的叶子上而且是最小值(负无穷)')
        print('练习6.5-5 略')
        print('练习6.5-6 先进先出队列和栈')
        A = [15, 13, 9, 5, 12, 8, 7, 4, 0, 6, 2, 1]
        print('练习6.5-7 数组A=', _deepcopy(A), '执行HEAP-DELETE(A, 2)操作的过程为：', heap.maxheapdelete(A, 2))
        print('练习6.5-8 略(不会)')
        print('思考题6-1 用插入的方法建堆')
        print('思考题6-2 对d叉堆的分析')
        print('思考题6-3 Young氏矩阵')
        print('')
        # python src/chapter6/chapter6_5.py
        # python3 src/chapter6/chapter6_5.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_5().note()
else:
    pass
