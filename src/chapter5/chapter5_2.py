
# python src/chapter5/chapter5_2.py
# python3 src/chapter5/chapter5_2.py
from __future__ import division, absolute_import, print_function
import sys as _sys

import math as _math

import random as _random

from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_2:
    '''
    CLRS 第五章 5.2 算法函数和笔记
    '''

    def __inversionListNum(self, array):

        # local function
        def __inversion(array, end):
            # 进行深拷贝保护变量
            list = _deepcopy([])
            n = _deepcopy(end)
            A = _deepcopy(array)
            if n > 1 :
                newList = __inversion(array, n - 1)
                # 相当于C#中的foreach(var x in newList); list.Append(x);
                for i in newList:
                    list.append(i)
            lastIndex = n - 1
            for i in range(lastIndex):
                if A[i] > A[lastIndex]:
                    list.append((i, lastIndex))
            return list

        return len(__inversion(array, len(array)))

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
        print('指示器随机变量为概率与期望之间的转换提供了一个便利的方法。')
        print('给定一个样本空间S和事件A，那么事件A对应的指示器随机变量I{A}的定义为')
        print('I(A)=1,如果A发生的话；I(A)=0,如果A不发生的话')
        print('引理5-1 给定样本空间S和S中的事件A，令Xa=I{A},则E[Xa]=Pr{A}')
        print('利用指示器随机变量分析雇佣问题')
        print('令X作为一个一个随机变量，其值等于雇佣一个新的办公助理的次数')
        print('特别地，令Xi对应于第i个应聘者被雇佣这个事件的指示器随机变量')
        print('Xi=I{第i位应聘者被雇佣}=1 or 0; 1代表被雇佣，0代表没有被雇佣')
        print('并且X=X1+X2+...+Xn')
        print('应聘者i比从应聘者i-1更有资格的概率是1/i,因此也以1/i的概率被雇佣(注意是1/i不是1/n)')
        print('重点：E[X]=sum(1/i)=lnn+O(1)')
        print('即使面试了n个人，平均看起来，实际上大约只雇佣他们之中的lnn个人')
        print('假设应聘者以随机的次序出现，算法HIRE-ASSISTANT总的雇佣费用为O(clnn)')
        print('练习5.2-1 正好雇佣一次的情况就是雇佣了最佳应聘者rank=n的情况，概率为1/n')
        print('练习5.2-2 正好雇佣两次的情况是第一个人除了rank=n那个人都可以，',
            '第二个人必须是最佳雇佣者 P=1/n*∑1/(n-i) ')
        print('练习5.2-3 掷一次骰子的期望数值是3.5，掷n次骰子的期望数值就是3.5n')
        print('练习5.2-4 (帽子保管问题)还回帽子的情况总共有n!种，每个人能拿到自己帽子的概率是1/n，',
            '期望值也是1/n,那么帽子总数的期望值就是每个人帽子数目期望值相加=1')
        print('练习5.2-5 排列的情况总共有n!中，最少情况是升序排列逆序对个数为0，', 
            '最多情况是降序排列n(n-1)/2')
        print('比如[6,5,4,3,2,1]的逆序对个数为6*5/2=15:', 
            self.__inversionListNum([6, 5, 4, 3, 2, 1]))
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
