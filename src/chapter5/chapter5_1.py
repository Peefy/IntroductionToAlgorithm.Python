
# python src/chapter5/chapter5_1.py
# python3 src/chapter5/chapter5_1.py
from __future__ import division, absolute_import, print_function
import sys
import math

from random import randint 

from copy import copy 
from copy import deepcopy 

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

import io
import sys 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 

class Chapter5_1:
    '''
    CLRS 第五章 5.1 算法函数和笔记
    '''

    def myRandom(self, a = 0, b = 1):
        '''
        产生[a,b]之间的随机整数
        '''
        return randint(a, b)

    def myBiasedRandom(self):
        pass

    def note(self):
        '''
        Summary
        =
        Print chapter5.1 note

        Example
        =
        >>> Chapter5_1().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.1 雇佣问题')
        print('假设需要雇佣一个一个新的办公室助理，之前的雇佣都失败了，所以决定找一个雇佣代理，雇佣代理每天推荐一个应聘者')
        print('每找到一个更好地应聘者，就辞掉之前的应聘者')
        print('当然雇佣，面试，开除，给代理中介费都需要一定的\"代价\"')
        print('HIRE-ASSISTANT(n)过程伪代码')
        print(' 1. best <- 0')
        print(' 2. for i <- 1 to n')
        print(' 3.     interview candidate i')
        print(' 4.     if candidate i is better than candidate best')
        print(' 5.          then best <- i')
        print(' 6.              hire candidate i')
        print('关心的重点不是HIRE-ASSISTANT的执行时间，而是面试和雇佣所花的费用')
        print('最坏情况分析')
        print('在最坏情况下，我们雇佣了每个面试的应聘者。当应聘者的资质逐渐递增时，就会出现这种情况，此时我们雇佣了n次，总的费用O(nc)')
        print('事实上既不能得知应聘者的出现次序，也不能控制这个次序。因此，通常我们预期的是一般或平均情况')
        print('概率分析是在问题的分析中应用概率技术，大多数情况下，使用概率分析来分析一个算法的运行时间')
        print('为了进行概率分析，必须使用关于输入分布的知识或对其假设，然后分析算法，计算出一个期望的运行时间')
        print('在所有应聘者的资格之间，存在一个全序关系。因此可以使用从1到n的唯一号码来讲应聘者排列名次')
        print('用rank(i)表示应聘者i的名次，并约定较高的名次对应较有资格的应聘者')
        print('这个有序序列rank(1),rank(2),...,rank(3)是序列1,2,...,n的一个排列')
        print('应聘者以随机的顺序出现，就等于说这个排名列表是数字1到n的n!(n的阶乘)')
        print('或者，也可以称这些排名构成一个均匀的随机排列；亦即在n!中可能的组合中，每一种都以相等的概率出现')
        print('随机算法：为了利用概率分析，需要了解关于输入分布的一些情况。在许多情况下，我们对输入分布知之甚少')
        print('一般的，如果一个算法的输入行为不只是由输入决定，同时也由随机数生成器所产生的数值决定，则称这个算法是随机的')
        print('练习5.1-1:每次HIRE应聘者是有一个顺序的，HIRE的时候同时把次序压如栈中，就得到了排名的总次序')
        random_list = [self.myRandom(), self.myRandom(), self.myRandom(), self.myRandom(), self.myRandom(),]
        print('产生5个[0,1]的随机整数', random_list)
        # python src/chapter5/chapter5_1.py
        # python3 src/chapter5/chapter5_1.py
        return self

_instance = Chapter5_1()
note = _instance.note  

if __name__ == '__main__':
    print('Run main : single chapter five!')
    Chapter5_1().note()
else:
    pass
