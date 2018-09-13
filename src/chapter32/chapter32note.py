# coding:utf-8
# usr/bin/python3
# python src/chapter32/chapter32note.py
# python3 src/chapter32/chapter32note.py
"""

Class Chapter32_1

Class Chapter32_2

Class Chapter32_3

Class Chapter32_4

"""
from __future__ import absolute_import, division, print_function

import math
import re
import numpy as np

if __name__ == '__main__':
    pass
else:
    pass

class Chapter32_1:
    """
    chapter32.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.1 note

        Example
        ====
        ```python
        Chapter32_1().note()
        ```
        """
        print('chapter32.1 note as follow')
        print('第32章 字符串匹配')
        print('在文本编辑程序中,经常出现要在一段文本中找出某个模式的全部出现位置这一问题。典型情况是,一段文本是正在编辑的文件,',
            '所搜寻的模式是用户提供的一个特定单词。解决这个问题的有效算法能极大地提高文本编辑程序的响应性能',
            '字符串匹配算法也常常用于其他方面,例如在DNA序列中搜寻特定的模式')
        print('字符串匹配问题的形式定义是这样的:假设文本是一个长度为n的数组T[1..n],模式是一个长度为m<=n的数组P[1..m].',
            '进一步假设P和T的元素都是属于有限字母表∑表中的字符.例如可以有∑={0,1}或∑={a,b,...,z},字符数组P和T常称为字符串')
        print('如果0<=s<=n-m,并且T[S+1,...,s+m]=P[1..m](即对1<=j<=m,有T[s+j]=P[j]),则说模式P在文本T中出现且位移为s.',
            '(或者等价地,模式P在文本T中从位置s+1开始出现)。如果P在T中出现且位移为s,则称s为一个有效位移,否则称s为无效位移',
            '这样一来,字符串匹配问题就变成一个在一段指定的文本T中,找出某指定模式P出现所有有效位移的问题')
        print('本章的每个字符串匹配算法都对模式进行了一些预处理,然后找寻所有有效位移;我们称第二步为“匹配”.',
            '每个算法的总运行时间为预处理和匹配时间的总和.')
        print('32.2节介绍由Rabin和Karp发现的一种有趣的字符串匹配算法,该算法在最坏情况下的运行时间为Θ((n-m+1)m),虽然这一时间并不比朴素的算法好',
            '但是在平均情况和实际情况中,该算法的效果要好的多.这种算法也可以很好地推广到解决其他的模式匹配问题')
        print('32.3节中描述另一种字符串匹配算法,该算法构造一个特别设计的有限自动机,用来搜寻某给定模式P在文本中的出现的位置',
            '此算法用O(m|∑|)的预处理时间,但只用Θ(n)的匹配时间')
        print('32.4节介绍与其类似但更巧妙的Knuth-Morris-Pratt(或KMP)算法。该算法的匹配时间同样为Θ(n),但是将预处理时间降至Θ(m)')
        print('算法          预处理时间        匹配时间')
        print('朴素算法          0           O((n-m+1)m)')
        print('Rabin-Karp       Θ(m)        O((n-m+1)m)')
        print('有限自动机算法   O(m|∑|)         Θ(n)')
        print('KMP算法          Θ(m)           Θ(n)')
        print('记号与术语')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_2:
    """
    chapter32.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.2 note

        Example
        ====
        ```python
        Chapter32_2().note()
        ```
        """
        print('chapter32.2 note as follow')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_3:
    """
    chapter32.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.3 note

        Example
        ====
        ```python
        Chapter32_3().note()
        ```
        """
        print('chapter32.3 note as follow')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

class Chapter32_4:
    """
    chapter32.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter32.4 note

        Example
        ====
        ```python
        Chapter32_4().note()
        ```
        """
        print('chapter32.4 note as follow')
        # python src/chapter32/chapter32note.py
        # python3 src/chapter32/chapter32note.py

chapter32_1 = Chapter32_1()
chapter32_2 = Chapter32_2()
chapter32_3 = Chapter32_3()
chapter32_4 = Chapter32_4()

def printchapter32note():
    """
    print chapter32 note.
    """
    print('Run main : single chapter thirty-two!')
    chapter32_1.note()
    chapter32_2.note()
    chapter32_3.note()
    chapter32_4.note()

# python src/chapter32/chapter32note.py
# python3 src/chapter32/chapter32note.py

if __name__ == '__main__':  
    printchapter32note()
else:
    pass
