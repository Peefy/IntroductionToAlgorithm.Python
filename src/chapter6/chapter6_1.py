
# python src/chapter6/chapter6_1.py
# python3 src/chapter6/chapter6_1.py
from __future__ import division, absolute_import, print_function
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange


import io
import sys 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 
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
        print('为研究比较排序算法性能的极限，第八章分析了决策树模型，通过非比较的方式进行排序,则可以突破Ω(nlgn)的下界')
        print('比如计数排序算法，基数排序算法')
        print('顺序统计学')
        print('在由n个数构成的集合上，第i个顺序统计是集合中第i个小的数')
        print('不必有高深的数学知识，但是需要特殊的数学技巧：快速排序，桶排序，顺序统计量悬法')
        print('第六章 堆排序')
        print('堆排序特点：复杂度Θ(nlgn)，原地(in place)排序，利用某种数据结构来管理算法当中的信息')
        print('堆这个词首先是在堆排序中出现，后来逐渐成为\“废料收集存储区\”')
        print('6.1 堆')
        print('(二叉)堆数据结构是一种数组对象，它被视为一颗完全二叉树，树的每一层都是填满的')
        print('表示堆的数组A是一个具有两个属性的对象，', 
            'length[A]是数组中的元素个数,heap-size[A]是存放在A中的堆的元素个数')
        print('虽然A[0..length(A)-1]中都可以包含有效值')
        print('但A[heap-size[A]]之后的元素都不属于相应的堆')
        print('此处length[A]>=heap-size[A],树的根为A[0],给定了某个结点的下标i，可以很轻松的求出其父节点，左儿子和右儿子的下标')
        print('比如下标i的父节点Parent为[i/2],左儿子Left为[2i],右儿子Right为[2i+1]')
        print('一个最大堆(大根堆)可被看作一个二叉树和一个数组')
        print('二叉堆有两种：最大堆和最小堆，最小堆的最小元素是在根部')
        print('最大堆：A[Parent[i]]>=A[i]')
        print('最小堆：A[Parent[i]]<=A[i]')
        print('在堆排序中，使用最大堆，最小堆通常在构造优先队列时使用')
        print('练习6.1-1：在高度为h的堆中，最少元素为1，最多元素为2^h')
        print('练习6.1-2：含n个元素的堆的高度为[lgn]')
        print('练习6.1-3：在一个最大堆的某颗子树，最大元素在该子树的根上')
        print('练习6.1-4：堆的最后的子树的子节点')
        print('练习6.1-5：一个升序排好的数组是一个最小堆')
        print('练习6.1-6：[23,17,14,6,13,10,1,5,7,12]是一个最大堆')
        print('练习6.1-7：当用数组表示了存储n个元素的堆时，叶子节点的下标[n/2]+1,[n/2]+2,...,n')
        # python src/chapter6/chapter6_1.py
        # python3 src/chapter6/chapter6_1.py
        return self

if __name__ == '__main__':  
    print('Run main : single chapter six!')  
    Chapter6_1().note()
else:
    pass
