
# python src/chapter3/chapter3_1.py
# python3 src/chapter3/chapter3_1.py

import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter3_1:
    '''
    CLRS 第三章 3.1 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter3.1 note

        Example
        =
        >>> Chapter3_1().note()
        '''
        print('第3章 函数的增长')
        print(' 对于足够大的输入规模，在精确表示的运行时间中，常数项和低阶项通常由输入规模所决定')
        print(' 当输入规模大到时只与运行时间的增长量级有关时，就是研究算法的渐进效率')
        print(' 从极限的角度看，只关心算法运行时间如何随着输入规模的无限增长而增长')
        print(' 对不是很小的输入规模而言，从渐进意义上说更有效的算法是最佳的选择')
        print('3.1渐进记号')
        print(' 算法最坏情况运行时间T(n),因为T(n)一般仅定义于整数的输入规模上')
        print(' Θ O Ω o ω五种记号 ') 
        print(' Θ记号：在第二章中，知道插入排序的最坏情况下运行时间是T(n)=Θ(n^2)')
        print(' 对一个给定的函数，用Θ(g(n))表示一个函数集合')
        print(' Θ记号渐进地给出一个函数的上界和下界(渐进确界)，当只有渐进上界时，使用O记号。对一个函数g(n),用O(g(n)表示一个函数集合)')
        print(' O记号是用来表示上界的，当用它作为算法的最坏情况运行时间的上界，就对任意输入有运行时间的上界')
        print(' 例子：插入排序在最坏情况下运行时间的上界O(n^2)也适用于每个输入的运行时间。')
        print(' 但是，插入排序最坏情况运行时间的界Θ(n^2)并不是对每种输入都适用。当输入已经排好序时，插入排序的运行时间为Θ(n)')
        print(' 正如O记号给出一个函数的渐进上界，Ω记号给出函数的渐进下界。给定一个函数g(n),用Ω(g(n))表示一个函数集合')
        print(' Ω记号描述了渐进下界，当它用来对一个算法最佳情况运行时间限界时，也隐含给出了在任意输入下运行时间的界。')
        print(' 例如：插入排序的最佳情况运行时间是Ω(n),隐含着该算法的运行时间是Ω(n)')
        print(' 定理3.1 对任意两个函数f(n)和g(n)，f(n)=Θ(g(n))当且仅当f(n)=O(g(n))和f(n)=Ω(g(n))')
        print(' 插入排序的运行时间介于Ω(n)和O(n^2)之间，因为它处于n的线性函数和二次函数的范围内')
        print(' 插入排序的运行时间不是Ω(n^2),因为存在一个输入(当输入已经排好序时)，使得插入排序的运行时间为Ω(n^2)')
        print(' 当说一个算法的运行时间(无修饰语)是Ω(g(n))时，是指对每一个n值，无论取该规模下什么样的输入，该输入上的运行时间都至少是一个常数乘上g(n)(当n足够大时)')
        print('等式和不等式中的渐进符号')
        print(' 合并排序的最坏情况运行时间表示为递归式：T(n)=2T(n/2)+Θ(n)')
        print(' 一个表达式中的匿名函数的个数与渐进记号出现的次数是一致的。')
        print(' 有时渐进记号出现在等式的左边，例如：2n^2+3n+1=2n^2+Θ(n)=Θ(n^2)')
        print('o记号:O记号所提供的渐进上界可能是也可能不是渐进紧确的。界2n^2=O(n^2)是渐进紧确的，但2n=O(n^2)却不是')
        print(' 使用o记号来表示非渐进紧确的上界，例如2n=o(n^2),但是2n^2≠o(n^2)')
        print('ω记号与Ω记号的关系就好像o记号与O记号的关系一样，用ω记号来表示非渐进紧确的下界')
        print('例如：n^2/2=ω(n),但n^2/2≠ω(n^2)')
        print('函数间的比较')
        print('假设f(n)和g(n)是渐进正值函数')
        # python src/chapter3/chapter3_1.py
        # python3 src/chapter3/chapter3_1.py
        
if __name__ == '__main__':
    print('Run main : single chapter three!')
    Chapter3_1().note()
else:
    pass
