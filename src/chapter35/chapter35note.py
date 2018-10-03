# coding:utf-8
# usr/bin/python3
# python src/chapter35/chapter35note.py
# python3 src/chapter35/chapter35note.py
"""

Class Chapter35_1

Class Chapter35_2

Class Chapter35_3

Class Chapter35_4

Class Chapter35_5

"""
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    pass
else:
    pass

class Chapter35_1:
    """
    chapter35.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.1 note

        Example
        ====
        ```python
        Chapter35_1().note()
        ```
        """
        print('chapter35.1 note as follow')
        print('第35章 近似算法')
        print('许多具有实际意义的问题都是NP完全问题,但都非常重要,所以不能仅因为获得其最优解的过程非常困难就放弃')
        print('解决NP完全问题至少有三种方法:第一,如果实际输入的规模比较小,则用具有指数运行时间的算法来解决问题就很理想了.',
            '第二,或许能将一些重要的、多项式时间可解的特殊情况隔离出来;第三,仍有可能在多项式时间里面找到(最坏情况或平均情况)近似最优解.',
            '在实践中,近似最优解常常就足够好了,就返回近似最优解的算法称为近似算法')
        print('近似算法的性能比值')
        print('假定在解一个最优化问题,该问题的每一个可能解都有正的代价,希望找出一个近似最优解.根据所要解决的问题',
            '最优解可以定义成具有最大可能代价的解或具有最小可能代价的解.就是说,该问题可能是一个求最大值的问题或求最小值的问题')
        print('说问题的一个近似算法有着近似比p(n),如果对规模为n的任何输入,由该近似算法产生的解的代价C与最优解的代价C*只差一个因子p(n)',
            'max(C/C*,C*/C)<=p(n),也称一个能达到近似比p(n)的算法为p(n)近似算法.这个定义对求最大值和求最小值问题都适用.',
            '对于一个求最大值的问题,0<C<=C*,而比值C*/C给出最优解的代价大于近似解的代价的倍数.类似地,对于求最小值问题也是同理')
        print('对于很多问题来说,已经设计出具有较小的固定近似比的多项式时间近似算法;对于另一些问题来说,在其已知的最佳多项式时间的近似算法中,',
            '近似比是作为输入规模n的函数而增长的')
        print('一些NP完全问题允许有多项式时间的近似算法,通过消耗越来越多的计算时间,这些近似算法可以达到不断缩小的近似比.就是说,在计算时间和近似的质量之间可以进行权衡')
        print('一个最优化问题的近似方案是这样的一种近似算法,它的输入除了该问题的实例外,还有一个值e>0,使得对任何固定的e,该方案是个(1+e)近似算法',
            '对一个近似方案来说,如果对任何固定的e>0,该方案都以其输入实例的规模n的多项式时间运行,则称此方案为多项式时间近似方案')
        print('随着e的减小,多项式时间近似方案的运行时间会迅速增长.例如,一个多项式时间近似方案的运行时间可能达到O(n^2/e).在理想情况下,',
            '如果e按一个常数因子减小,为了获得希望的近似效果,所增加的运行时间不应该超过一个常数因子.',
            '希望运行时间既是1/e的多项式,又是n的多项式')
        print('对一个近似方案来说,如果其运行时间既是1/e的多项式,又为输入实例的规模n的多项式,则称其为完全多项式时间的近似方案.',
            '例如,近似方案可能有O((1/e)^2n^3)运行时间.对于这样的一种方案,e的任意常数倍的减少可以由运行时间的相应常数倍增加来弥补')
        print('35.1 顶点覆盖问题')
        print('虽然在一个图G中寻找最优顶点覆盖比较困难,但要找出一个近似最优的顶点覆盖不会太难.下面给出的近似算法以一个无向图G为输入,',
            '并返回一个其规模保证不超过最优顶点覆盖的规模两倍的顶点覆盖规模两倍的顶点覆盖')
        print('定理35.1 APPROX-VERTEX-COVER有一个多项式时间的2近似算法')
        print('练习35.1-1 ')
        print('练习35.1-2 ')
        print('练习35.1-3 ')
        print('练习35.1-4 ')
        print('练习35.1-5 ')
        print('')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_2:
    """
    chapter35.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.2 note

        Example
        ====
        ```python
        Chapter35_2().note()
        ```
        """
        print('chapter35.2 note as follow')
        print('35.2 旅行商问题')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_3:
    """
    chapter35.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.3 note

        Example
        ====
        ```python
        Chapter35_3().note()
        ```
        """
        print('chapter35.3 note as follow')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_4:
    """
    chapter35.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.4 note

        Example
        ====
        ```python
        Chapter35_4().note()
        ```
        """
        print('chapter35.4 note as follow')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

class Chapter35_5:
    """
    chapter35.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter35.5 note

        Example
        ====
        ```python
        Chapter35_5().note()
        ```
        """
        print('chapter35.5 note as follow')
        # python src/chapter35/chapter35note.py
        # python3 src/chapter35/chapter35note.py

chapter35_1 = Chapter35_1()
chapter35_2 = Chapter35_2()
chapter35_3 = Chapter35_3()
chapter35_4 = Chapter35_4()
chapter35_5 = Chapter35_5()

def printchapter35note():
    """
    print chapter35 note.
    """
    print('Run main : single chapter thirty-five!')
    chapter35_1.note()
    chapter35_2.note()
    chapter35_3.note()
    chapter35_4.note()
    chapter35_5.note()

# python src/chapter35/chapter35note.py
# python3 src/chapter35/chapter35note.py

if __name__ == '__main__':  
    printchapter35note()
else:
    pass
