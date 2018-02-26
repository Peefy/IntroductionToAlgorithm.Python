
# python src/chapter4/chapter4_2.py
# python3 src/chapter4/chapter4_2.py

import sys
import math

from copy import copy
from copy import deepcopy

import numpy as np
from numpy import arange

from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

class Chapter4_2:
    '''
    CLRS 第四章 4.2 算法函数和笔记
    '''
    def note(self):
        '''
        Summary
        =
        Print chapter4.2 note

        Example
        =
        >>> Chapter4_2().note()
        '''
        print('4.2 递归树方法')
        print('虽然代换法给递归式的解的正确性提供了一种简单的证明方法，但是有的时候很难得到一个好的猜测')
        print('就像分析合并排序递归式那样，画出一个递归树是一种得到好猜测的直接方法。在递归树中，每一个结点都代表递归函数调用集合中的一个子问题的代价')
        print('将树中每一层内的代价相加得到一个每层代价的集合，再将每层的代价相加得到递归是所有层次的总代价')
        print('当用递归式表示分治算法的运行时间时，递归树的方法尤其有用')
        print('递归树最适合用来产生好的猜测，然后用代换法加以验证。')
        print('但使用递归树产生好的猜测时，通常可以容忍小量的不良量，因为稍后就会证明')
        print('如果画递归树时非常的仔细，并且将代价都加了起来，那么就可以直接用递归树作为递归式解的证明')
        print('建立一颗关于递归式 T(n)=3T(n/4)+cn^2 的递归树;c>0,为了方便假设n是4的幂，根部的cn^2项表示递归在顶层时所花的代价')
        print('如果递归树代价准确计算出就可以直接作为解，如果不准确计算出代价就可以为代换法提供一个很好的假设')
        print('不准确计算出递归树代价的一个例子： T(n)=T(n/3)+T(2n/3)+O(n)')
        print('为了简化起见，此处还是省略了下取整函数和上取整函数，使用c来代表O(n)项的常数因子。当将递归树内各层的数值加起来时，可以得到每一层的cn值')
        print('从根部到叶子的最长路径是n->(2/3)n->(2/3)^2n->...->1')
        print('因为当k=log3/2(n)时，(2/3)^kn=1,所以树的深度是log3/2(n)')
        print('如果这颗树是高度为log3/2(n)的完整二叉树，那么就有2^(log3/2(n))=n^(log3/2(2))个叶子')
        print('由于叶子代价是常数，因此所有叶子代价的总和为Θ(n^(log3/2(2))),或者说ω(nlgn)')
        print('然而，这颗递归树并不是完整的二叉树，少于n^(log3/2(2))个叶子，而且从树根往下的过程中，越来越多的内部节点在消失')
        print('因此，并不是所有层次都刚好需要cn代价；越靠近底层，需要的代价越少')
        print('虽然可以计算出准确的总代价，但记住我们只是想要找出一个猜测来使用到代换法中')
        print('容忍这些误差，而来证明上界O(nlgn)的猜测是正确的')
        print('事实上，可以用代换法来证明O(nlgn)是递归式解的上界。下面证明T(n)<=dnlgn,当d是一个合适的正值常数')
        print('T(n)<=T(n/3)+T(2n/3)+cn<=dnlgn;成立条件是d>=c/(lg3-(2/3)),因此没有必要准确地计算递归树中的代价')
        print('练习4.2-1:树的深度为lgn,等比数列求和公式为S=a1(1-q^n)/(1-q),所以树的总代价为dn^2+Θ(n^lg3)')
        print('练习4.2-2:书中已经证明过了')
        print('练习4.2-3:')
        print('练习4.2-4:')
        print('练习4.2-5:')
        # python src/chapter4/chapter4_2.py
        # python3 src/chapter4/chapter4_2.py
        
if __name__ == '__main__':
    print('Run main : single chapter four!')
    Chapter4_2().note()
else:
    pass
