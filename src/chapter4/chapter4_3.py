
# python src/chapter4/chapter4_1.py
# python3 src/chapter4/chapter4_1.py

import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_3:
    '''
    CLRS 第四章 4.3 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.3 note

        Example
        =
        >>> Chapter4_3().note()
        '''
        print('4.3 主方法')
        print('主方法给出求解形如 T(n) = aT(n/b)+f(n) 的递归式子的解，其中a>=1,b>1,f(n)是一个渐进正的函数')
        print('主方法要求记忆三种情况，这样可以很容易确定许多递归式子的解')
        print('递归式描述了将规模为n的问题划分为a个子问题的算法的运行时间，每个子问题规模为n/b,a和b是正常数')
        print('a个子问题被分别递归地解决，时间各为T(n/b)。划分原问题和合并答案的代价由函数f(n)描述')
        print('如合并排序过程的递归式中有a=2;b=2,f(n)=Θ(n)')
        print('或者将递归式写为T(n) = aT([n/b])+f(n)')
        print('定理4.1(主定理)')
        print(' 1.若对于某常数ε>0,有f(n)=O(n^(logb(a)-ε)),则T(n)=Θ(n^(logb(a)))')
        print(' 2.若f(n)=Θ(n^logb(a)),则T(n)=Θ(n^(logb(a))lgn)')
        print(' 3.若对于某常数ε>0,有f(n)=Ω(n^(logb(a)+ε)),且对常数c<1与所有足够大的n,有af(n/b)<=cf(n),则T(n)=Θ(f(n))')
        print('以上三种情况，都把函数f(n)与函数n^logb(a)进行比较,1中函数n^logb(a)更大，则解为T(n)=Θ(n^(logb(a)))')
        print('而在3情况中，f(n)是较大的函数，则解为Θ(f(n)),在第二种情况中函数同样大，乘以对数因子，则解为T(n)=Θ(n^(logb(a))lgn)')
        print('但是三种情况并没有覆盖所有可能的f(n),如果三种情况都满足，则主方法不能用于解递归式子')
        print('主方法的应用')
        print(' 1.T(n)=9(n/3)+n,对应于主定理中第一种情况T(n)=Θ(n^2)')
        print(' 2.T(n)=(2n/3)+1,对应于主定理中第二种情况T(n)=Θ(lgn)')
        print(' 3.T(n)=3T(n/4)+nlgn,有a=3,b=4,f(n)=nlgn,对应于主定理中第三种情况T(n)=Θ(nlgn)')
        print(' 4.递归式T(n)=2T(n/2)+nlgn对主定理方法不适用 nlgn 渐进大于n，并不是多项式大于，所以落在情况二和情况三之间')
        print('练习4.3-1 a) T(n)=Θ(n^2); b) T(n)=Θ(n^2lgn); c) T(n)=Θ(n^2);')
        print('练习4.3-2 算法A的运行时间解由主定理求得T(n)=Θ(n^2),a最大整数值为16')
        print('练习4.3-3 属于主定理的第二种情况T(n)=Θ(lgn)')
        print('练习4.3-4 不能用主定理给出渐进确界')
        print('练习4.3-5 略')
        # python src/chapter4/chapter4_3.py
        # python3 src/chapter4/chapter4_3.py
        return self
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_3().note()
else:
    pass
