
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

class Chapter4_1:
    '''
    CLRS 第四章 4.1 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.1 note

        Example
        =
        >>> Chapter4_1().note()
        '''
        print('第四章 递归式')
        print('当一个算法包含对自身的递归调用时，其运行时间通常可以用递归式来表示，递归式是一种等式或者不等式')
        print('递归式所描述的函数是用在更小的输入下该函数的值来定义的')
        print('本章介绍三种解递归式子的三种方法,找出解的渐进界Θ或O')
        print('1.代换法：先猜测某个界存在，再用数学归纳法猜测解的正确性')
        print('2.递归树方法：将递归式转换为树形结构')
        print('3.主方法：给出递归形式T(n)=aT(n/b)+f(n),a>=1;b>1;f(n)是给定的函数')
        print('4.1 代换法')
        print('代换法解递归式的两个步骤，代换法这一名称源于当归纳假设用较小值时，用所猜测的值代替函数的解',
            '，这种方法很有效，但是只能用于解的形式很容易猜的情形')
        print(' 1.猜测解的形式')
        print(' 2.用数学归纳法找出使解真正有效的常数')
        print('代换法可用来确定一个递归式的上界或下界。')
        print('例子：确定递归式 T(n)=2T([n/2])+n 的一个上界，首先猜测解为T(n)=O(nlgn),然后证明T(n)<cnlgn (上界的定义);c>0')
        print('假设这个界对[n/2]成立，即T([n/2])<=c[n/2]lg([n/2])<=cnlg(n/2)+n=cnlgn-cnlg2+n=cnlgn-cn+n<=cnlgn')
        print('最后一步只要c>=1就成立')
        print('接下来应用数学归纳法就要求对边界条件成立。一般来说，可以通过证明边界条件符合归纳证明的基本情况来说明它的正确性')
        print('对于递归式 T(n)=2T([n/2])+n ，必须证明能够选择足够大的常数c, 使界T(n)<=cnlgn也对边界条件成立')
        print('假设T(1)=1是递归式唯一的边界条件。那么对于n=1时，界T(n)<=cnlgn也就是T(1)<=c1lg1=0,与T(1)=1不符')
        print('因此，归纳证明的基本情况不能满足')
        print('对特殊边界条件证明归纳假设中的这种困难很容易解决。对于递归式 T(n)=2T([n/2])+n ,利用渐进记号，只要求对n>=n0,证明T(n)<=cnlgn,其中n0是常数')
        print('大部分递归式，可以直接扩展边界条件，使递归假设对很小的n也成立')
        print('不幸的是，并不存在通用的方法来猜测递归式的正确解，猜测需要经验甚至是创造性的')
        print('例如递归式 T(n) = 2T([n/2]+17)+n 猜测T(n)=O(nlgn)')
        print('猜测答案的另一种方法是先证明递归式的较松的上下界，因为递归式中有n，而我们可以证明初始上届为T(n)=O(n^2)')
        print('然后逐步降低其上界，提高其下界，直至达到正确的渐进确界T(n)=Θ(nlgn)')
        print('例子：T(n)=T([n/2])+T([n/2])+1 ')
        print('先假设解为T(n)=O(n),即要证明对适当选择的c，有T(n)<=cn')
        print('T(n)<=c[n/2]+c[n/2]+1=cn+1,但是无法证明T(n)<=cn,所以可能会猜测一个更大的界,如T(n)=O(n^2),当然也是一个上界')
        print('当然正确的解是T(n)=O(n)')
        print('避免陷阱：在运用渐进表示时很容易出错，例如T(n)<=2(c[n/2])+n<=cn+n=O(n),因为c是常数，因而错误地证明了T(n)=O(n)')
        print('错误在与没有证明归纳假设的准确形式')
        print('变量代换：有时对一个陌生的递归式作一些简单的代数变换，就会使之变成熟悉的形式，考虑T(n)=2T([sqrt(n)])+lgn,令m=lgn即可')
        print('练习4.1-1：使用代换法，假设不等式成立，即T(n)<=clgn;c>0，当然也有T([n/2])<=clg([n/2])')
        print(' 所以T(n)=T([n/2])+1<=clgn-c+1,所以只要c取的足够大就能使n>=2均使不等式成立，得证')
        print('练习4.1-2：使用代换法，假设不等式成立，即T(n)<=clgn;c>0，当然也有T([n/2])<=clg([n/2])')
        print(' T(n)=2T([n/2])+n<=cnlgn-cnlg2+n=cnlgn-cn+n<=cnlgn;当且仅当c>=1成立')
        print(' 再使用代换法证明T(n)>=clgn;c>0, T(n)=2T([n/2])+n>=cnlgn-cnlg2+n=cnlgn-cn+n>=cnlgn;当且仅当c<1时成立')
        print(' 所以递归的解的确定界为Θ(nlgn)')
        print('练习4.1-3：略')
        print('练习4.1-4：略')
        print('练习4.1-5：略')
        print('练习4.1-6：令m=lgn')
        # python src/chapter4/chapter4_1.py
        # python3 src/chapter4/chapter4_1.py
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_1().note()
else:
    pass
