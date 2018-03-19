
# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
'''
Class Chapter8_1

Class Chapter8_2

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

class Chapter8_1:
    '''
    chpater8.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter8.1 note

        Example
        ====
        ```python
        Chapter8_1().note()
        ```
        '''
        print('chapter8.1 note as follow')
        print('第8章 线性时间排序')
        print('合并排序和堆排序在最坏情况下能达到O(nlgn),快速排序在平均情况下达到此上界面')
        print('第8章之前的排序算法都是比较排序')
        print('8.1节中将证明对含有n个元素的一个输入序列，', 
            '任何比较排序在最坏情况下都要用Ω(nlgn)次比较来进行排序')
        print('由此可知，合并排序和堆排序是最优的')
        print('本章还介绍三种线性时间排序,计数排序，基数排序，桶排序')
        print('8.1 排序算法运行时间的下界')
        print('决策树模型')
        print('比较排序可以被抽象地看作决策树，一颗决策树是一个满二叉树')
        print('在决策树中，每个节点都标有i：j，其中1<=i,j<=n,n是输入序列中元素的个数，控制结构，数据移动等都被忽略')
        print('如排序算法的决策树的执行对应于遍历从树的根到叶子节点的路径')
        print('要使排序算法能正确的工作，其必要条件是n个元素n！种排列中的每一种都要作为一个叶子出现')
        print('对于根结点来说，每一个叶子都可以是某条路径可以达到的')
        print('比较排序算法最坏情况下界，就是从根部到最底部叶子走过的最长路径，也就是树的高度nlgn')
        print('定理8.1 任意一个比较排序在最坏情况下，都需要做Ω(nlgn)次的比较')
        print('堆排序和合并排序都是渐进最优的比较排序算法,运行时间上界O(nlgn)')
        print('练习8.1-1 最小深度可能是n-1，对于已经n个排序好的元素比较n-1次即可，如三个元素比较两次')
        print('练习8.1-2 斯特林近似公式是求n！的一个近似公式')
        print('练习8.1-3 对于长度为n的n!种输入，至少一半而言，不存在线性运行时间的比较排序算法')
        print('练习8.1-4 现有n个元素需要排序，它包含n/k个子序列，每一个包含n个元素')
        print(' 每个子序列的所有元素都小于后续序列的所有元素，所以对n/k个子序列排序，就可以得到整个输入长度的排序')
        print(' 这个排序问题中所需的问题都需要有一个下界Ω(nlgk)')
        # python src/chapter8/chapter8note.py
        # python3 src/chapter8/chapter8note.py

class Chapter8_2:
    '''
    chpater8.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter8.2 note

        Example
        ====
        ```python
        Chapter8_2().note()
        ```
        '''
        print('chapter8.2 note as follow')

chapter8_1 = Chapter8_1()
chapter8_2 = Chapter8_2()

def printchapter8note():
    '''
    print chapter8 note.
    '''
    print('Run main : single chapter eight!')  
    chapter8_1.note()
    chapter8_2.note()

# python src/chapter8/chapter8note.py
# python3 src/chapter8/chapter8note.py
if __name__ == '__main__':  
    printchapter8note()
else:
    pass
