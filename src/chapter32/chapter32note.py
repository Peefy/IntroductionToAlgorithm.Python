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

import stringmatch as sm

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
        print('  用∑*表示用字母表∑中的字符形成的所有有限长度的字符串的集合.在本章中仅考虑长度有限的字符串',
            '长度为0的空字符串用e表示,它也属于∑*.字符串x的长度用|x|表示.两个字符串x和y的连接表示为xy,其长度为|x|+|y|,由x的字符接y的字符组成')
        print('  如果对某个字符串与y∈∑*,有x=wy,就说字符串w是字符串x的前缀,表示为w∝x,得知|w|<=|x|.空字符串e既是每个字符串的前缀,也是每个字符串的后缀',
            '例如,有ab>abcca,cca<abcca.对任意字符串x和y及任意字符a,x>y当且仅当xa>ya.注意>和<都是传递关系')
        print('引理32.1(重叠后缀定理)假设x,y和z是满足x>z和y<z的三个字符串.如果|x|<=|y|,则x>y;如果|x|>=|y|,则y>x;如果|x|=|y|,则x=y')
        print('  本章中允许把比较两个等长的字符串是否相等的操作当做原语操作.如果对字符串的比较是从左往右进行,并且发现一个不匹配字符时比较就终止,',
            '则假设这样一个测试过程所需的时间是关于所发现的匹配字符数目的线性函数')
        print('32.1 朴素的字符串匹配算法')
        print('朴素的字符串匹配算法:它用一个循环来找出所有有效位移,该循环对n-m+1可能的每一个s值检查条件P[1..m]=T[s+1..s+m]')
        print('这种朴素的字符串匹配过程可以形象地看成用一个包含模式的“模板”沿文本滑动,同时对每个位移注意模板上的字符是否与文本的相应字符相等')
        print('NATIVE-STRING-MATCHER的运行时间为Θ((m-m+1)m)')
        print('在本章中还要介绍一种算法,它的最坏情况预处理时间为Θ(m),最坏情况匹配时间为Θ(n)')
        print('练习31.1-1 解答过程如下：')
        P = '0001'
        T = '000010001010001'
        sm.native_string_matcher(T, P)
        print('练习31.1-2 假设模式P中的所有字符都是不同的。试说明如何对一段n个字符的文本T加速过程NATIVE-STRING-MATCHER的执行速度,',
            '使其运行时间达到O(n)')
        print('练习31.1-3 假设模式P和文本T是长度分别为m和n的随机选取的字符串,其字符串属于d个元素的字母表∑={0,1,...,d-1},其中d>=2',
            '证明朴素算法第4行中隐含的循环所执行的字符比较的预计次数为')
        print('   (n-m+1)(1-d**-m)/(1-d**-1)<=2(n-m+1)')
        print('练习31.1-4 假设允许模式P中包含一个间隔字符◇,该字符可以与任意的字符串匹配(甚至可以与长度为0的字符串匹配)',
            '例如,模式ab◇ba◇c')
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
        print('Rabin-Karp算法')
        print('在实际应用中,Rabin和Karp所建议的字符串匹配算法能够较好的运行,我们还可以从中归纳出有关问题的其他算法,如二维模式匹配')
        print('Rabin-Karp算法预处理时间为')
        print('Θ(m),在最坏情况下的运行时间为O((n-m+1)m),但是它的平均情况运行时间还是比较好的')
        print('假定∑={0,1,2,...,9},这样每个字符都是一个十进制数字(一般情况下,可以假定每个字符都是基数为d的表示法中一个数字,d=|∑|).',
            '可以用一个长度为k的十进制数来表示由k个连续字符组成的字符串.因此,字符串31415就对应于十进制数31415',
            '如果输入字符既可以看做图形符号,也可以看做数字')
        print('已知一个模式P[1..m],设p表示其相应的十进制数的值。对于给定的文本T[1..n],用ts来表示其长度为m的子字符串T[s+1..s+m](s=0,1,...,n - m)相应的十进制数的值')
        print('当然,用ts来表示其长度为m的子字符串T[s+1..s+m](s=0,1,..,n-m)相应十进制数的值.当然,ts=p当且仅当T[s+1..s+m]=P[1..m],因此s是有效位移当且仅当ts=p',
            '如果能够在Θ(m)的时间内计算出p的值,并在总共Θ(n-m+1)的时间内计算出所有ts的值,那么通过把p值与每个ts值进行比较,能够在Θ(n)的时间内,求出有效位移s')
        print('可以运用霍纳法则,在Θ(m)的时间内计算出p的值：')
        print('  p=P[m]+10(P[m-1]+10(P[m-2]+...+10(P[2]+10P[1])...))')
        print('类似地,也可以在Θ(m)的时间内,根据T[1..m]计算出t0的值')
        print('为了在Θ(n-m)的时间内计算出剩余的值t1,t2,...,tn-m,可以在常数时间内根据ts计算出ts+1,这是因为霍纳法则')
        print('')
        print('')
        print('')
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
