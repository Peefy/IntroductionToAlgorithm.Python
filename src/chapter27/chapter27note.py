# coding:utf-8
# usr/bin/python3
# python src/chapter27/chapter27note.py
# python3 src/chapter27/chapter27note.py
'''

Class Chapter27_1

Class Chapter27_2

Class Chapter27_3

Class Chapter27_4

Class Chapter27_5

'''
from __future__ import absolute_import, division, print_function

import math as _math
import random as _random
import time as _time
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from random import randint as _randint

import numpy as np
from numpy import arange as _arange
from numpy import array as _array

if __name__ == '__main__':
    pass
else:
    pass

class Chapter27_1:
    '''
    chpater27.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.1 note

        Example
        ====
        ```python
        Chapter27_1().note()
        ```
        '''
        print('chapter27.1 note as follow')  
        print('第七部分 算法研究问题选编')
        print('第27章给出一种并行计算模型,即比较网络.比较网络是允许同时进行很多比较的一种算法',
            '可以建立比较网络,使其在O(lg^2n)运行时间内对n个数进行排序')
        print('第28章研究矩阵操作的高效算法,通过考察矩阵的一些基本性质,讨论Strassen算法',
            '可以在O(n^2.81)时间内将两个n*n矩阵相乘.然后给出两种通用算法,即LU分解和LUP分解,',
            '在利用高斯消去法在O(n^3)时间内解线性方程时要用到这两种方法',
            '当一组线性方程没有精确解时,如何计算最小二乘近似解')
        print('第29章研究线性规划.在给定资源限制和竞争限制下,希望得到最大或最小的目标',
            '线性规划产生于多种实践应用领域.单纯形法')
        print('第30章 快速傅里叶变换FFT,用于在O(nlgn)运行时间内计算两个n次多项式的乘积')
        print('第31章 数论的算法：最大公因数的欧几里得算法；',
            '求解模运算的线性方程组解法，求解一个数的幂对另一个数的模的算法',
            'RSA公用密钥加密系统，Miller-Rabin随机算法素数测试,有效地找出大的素数；整数分解因数')
        print('第32章 在一段给定的正文字符串中，找出给定模式的字符串的全部出现位置')
        print('第33章 计算几何学')
        print('第34章 NP完全问题')
        print('第35章 运用近似算法有效地找出NP完全问题的近似解')
        print('第27章 排序网络')
        print('串行计算机(RAM计算机)上的排序算法,这类计算机每次只能执行一个操作',
            '本章中所讨论的排序算法基于计算的一种比较网络模型','这种网络模型中,可以同时执行多个比较操作')
        print('比较网络与RAM的区别主要在于两个方面.前者只能执行比较,因此,像计数排序这样的算法就不能在比较网络上实现',
            '其次,在RAM模型中,各操作是串行执行的,即一个操作紧接着另一个操作')
        print('在比较玩过中,操作可以同时发生,或者以并行方式发生,这一特点使得我们能够构造出一种在次线性的运行时间内对n个值进行排序的比较网络')
        print('27.1 比较网络')
        print('排序网络总是能对其他输入进行排序的比较网络,比较网络仅由线路和比较器构成',
            '比较器是具有两个输入x和y以及两个输出x`和y`的一个装置,它执行下列函数')
        print('假设每个比较器操作占用的时间为O(1),换句话说,假定出现输入值x和y与产生输出值x`和y`之间的时间为常数')
        print('一条线路把一个值从一处传输到另一处,可以把一个比较器的输出端与另一个比较器的输入端相连',
            '在其他情况下,它要么是网络的输入线,要么是网络的输出线.',
            '在本章中都假定比较网络含n条输入线以及n条输出线')
        print('只有当同时有两个输入时,比较器才能产生输出值.假设在时间0输入线路上出现了一个输入序列<9,5,2,6>',
            '则在时刻0，只有比较器A和B同时存在两个输入值.假定每个比较器要花1个单位的时间来计算出输出值')
        print('在每个比较器均运行单位时间的假设下,可以对比较网络的\"运行时间\"作出定义',
            '就是从输入线路接受到其值的时刻,到所有输出线路收到其值所花费的时间.',
            '非形式地说,这一运行时间就是任何输入元素从输入线路到输出所经过的比较器数目的最大值')
        print('一条线路的深度可以定义：比较网络的输入线路深度为0.如果一个比较器有两条深度分别为dx和dy的输入线路',
            '则其输出线路的深度为max(dx+dy)+1')
        print('由于比较网络中没有比较器回路,所以线路的深度有明确定义,并且定义比较器的深度为其输出线路的深度')
        print('排序网络是指对每个输入序列,其输出序列均为单调递增(即b1<=b2<=...<=bn)的一种比较网络')
        print('比较网络与过程的相似之处在于它指定如何进行比较,其不同之处在于其实际规模决定于输入和输出的数目')
        print('练习27.1-1 给定一输入序列<9 6 5 2>,说明图上网络所有线路出现的值')
        print('练习27.1-2 设n为2的幂，试说明如何构造一个具有n个输入和n个输出，且深度为lgn的比较网络，',
            '其顶部的输出线路总是输出最小的输入值，而底部的输出线路则总是输出最大的输入值')
        print('练习27.1-3 向一个比较器后,所得的比较网络可能不再是排序网络了')
        print('练习27.1-4 证明任何具有n个输入的排序网路的深度至少为lgn')
        print('练习27.1-5 证明任何排序网络中的比较器的数目至少为Ω(nlgn)')
        print('练习27.1-6 说明排序网络的结构与插入排序有何关系')
        print('练习27.1-7 可以把C个比较器和n个输入的比较网络表示为取值范围从1到n,c对整数组成的一张表',
            '如果两对整数中包含同一整数,则在网络中相应的比较器排序由整数对的次序决定',
            '并描述一个运行时间为O(n+C)的串行算法来计算比较网络的深度')
        print('练习27.1-8 颠倒型比较器,这种比较器在其底部线路中产生最大输出值',
            '试说明如何把c个标准或颠倒的比较器组成的任意网络,转换为仅包含c个标准比较器的排序网络,证明所给出的转换方法是正确的')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_2:
    '''
    chpater27.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.2 note

        Example
        ====
        ```python
        Chapter27_2().note()
        ```
        '''
        print('chapter27.2 note as follow')  
        print('27.2 0-1原理')
        print('0-1原理认为，对于属于集合{0,1}的每个输入值,排序网络都能正确运行,则对任意输入值,它也能正确运行',
            '当构造排序网络和其他比较网络时,0-1原理使把注意力集中于对由0和1时组成的输入序列进行相应的操作',
            '一旦构造好排序网络,并证明它能对所有的0-1序列进行排序时,就可以运用0-1原理,说明他能对任意值序列进行正确的排序')
        print('引理27.1 如果比较网络把输入序列a=<a1,a2,a3,...,an>转化为输入序列b=<b1,b2,...,bn>',
            '则对任意单调递增函数f，该网络把输入序列f(a)=<f(a1),f(a2),...,f(an)>,转化为输出序列',
            'f(b)=<f(b1),f(b2),...,f(bn)>')
        print('对一般的比较网络中每条线路的深度进行归纳，从而证明一个比上述引理更强的结论：',
            '当把序列a作为网络的输入时，如果每条线路的值为ai，则把序列f(a)作为网络的输入时该线路的值为f(ai)',
            '因为输出线路包含于上述结论中，所以证明了该结论，也就证明了引理')
        print('定理27.2(0-1原理) 如果一个具有n个输入的比较网络能够对所有可能存在的2^n个0和1组成的序列进行正确的排序',
            '则对所有任意数组成的序列,该比较网络也可能对其正确的排序')
        print('练习27.2-1 证明：把一个单调递增函数作用于一个已排序序列后，得到的仍然是一个排序序列')
        print('练习27.2-2 证明：当且仅当能正确地对如下n-1个0-1序列进行排序：<1,0,0,...,0,0>,<1,1,0,...,0,0>',
            ',...,<1,1,1,...,1,0>,具有n个输入的比较网络才能够正确地对输入序列<n,n-1,...,1>进行排序')
        print('练习27.2-3 运用0-1原则，证明图27-6所示的比较网络为一个排序网络')
        print('练习27.2-4 对判定树模型(decision-tree model)阐述并证明与0-1原理类似的结论(提示：要正确地处理等式)')
        print('练习27.2-5 证明：对所有i=1,2,...,n-1,在一个具有n个输入的排序网络中,第i条线与第i+1条线之间必至少有一个比较器')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_3:
    '''
    chpater27.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.3 note

        Example
        ====
        ```python
        Chapter27_3().note()
        ```
        '''
        print('chapter27.3 note as follow')  
        print('27.3 双调排序网络')
        print('要构造有效的排序网络，第一步是构造一个能对任意双调序列(bitonic sequence)进行的比较网络')
        print('双调序列是指序列要么先单调递增后再单调递减，或者循环移动成为先单调递增后再单调递减')
        print('例如序列<1,4,6,8,3,2>,<6,9,4,2,3,5>和<9,8,3,2,4,6>都是双调的')
        print('对于边界情况,任何一个1个和2个数的序列都是双调序列。双调的0-1序列的结构比较简单,',
            '其形式为0^i 1^j 0^k或1^i 0^j 1^k,其中i,j,k>=0.必须注意单调递增或单调递减的序列也是单调的')
        print('将要构造的双调排序程序是一个能对0和1的双调序列进行排序的比较网络',
            '双调排序程序可以对任意数组成的双调序列进行排序')
        print('半清洁器')
        print('  双调排序由一些阶段组成,其中每一个阶段称为一个半清洁器(half-cleaner).',
            '每个半清洁器是一个深度为1的比较网络,其中输入线i与输出线i+n/2进行比较,i=1,2,...,n/2(假设n为偶数)')
        print('  当由0和1组成的双调序列作用于半清洁器输入时,半清洁器产生一个满足下列条件的输出序列:较小的值位于输出的上半部,较大的值位于输出的下半部',
            '并且两部分序列仍然是双调的。')
        print('  事实上，两部分序列中至少有一部分是清洁的--全由0或全由1组成。正是由于这一性质，才称其为\"半清洁器\",')
        print('引理27.3 如果半清洁器的输入是一个由0和1组成的双调序列，则其输出满足如下性质：输出的上半部分与下半部分都是双调的',
            '上半部分输出的每一个元素与下半部分输出的每个元素一样小,并且两部分中至少有一个部分是清洁的')
        print('双调排序器')
        print('  通过递归地连接半清洁器，就可以建立一个双调排序器，它是一个对双调序列进行排序的网络。',
            'BITONIC-SORTER[n]的第一个阶段由HALF-CLEANER[n]组成.由引理27.3可知,HALF-CLEANER[n]产生两个规模缩小一半的双调序列,',
            '且满足上半部分的每个元素至少与下半部分的每个元素一样小。因此，可以运用两个BITONIC-SORTER[n/2]分别对两部分递归地进行排序,从而完成整个排序工作')
        print('  BITONIC-SORTER[n]的深度D(n)由下列递归式给出:')
        print('  D(n)=0 if n == 1; D(n)=D(n/2)+1 if n == 2^k and k >= 1')
        print('  可以推得其解为D(n)=lgn')
        print('因此，可以用BITONIC-SORTER对深度为lgn的0-1双调序列进行排序','由类似于0-1原理的结论可知：',
            '该网络能对由任意数组成的双调序列进行排序')
        print('练习27.3-1 n=1,存在1个;n=2时存在2个;n=3时存在2个;n=4时存在6个;n=5时存在12个')
        print('  结论存在m个由0和1组成的双调序列 m=n if n <= 2; m=(n-1)(n-2) if n >= 3')
        print('练习27.3-2 证明当n为2的幂时,BITONIC-SORTER[n]包含Θ(nlgn)个比较器')
        print('练习27.3-3 说明当输入数n不是2的幂时,如何构造一个深度为O(lgn)的双调排序器')
        print('练习27.3-4 如果某半清洁器的输入是一个由任意数组成的双调序列,证明输出端满足下列性质:',
            '输出的上半部分和下半部分都是双调的,上半部分中的每个元素至少与下半部分中的每个元素一样小')
        print('练习27.3-5 考察两个由0和1组成的序列.证明如果其中一个序列的每个元素至少和另一个序列中每个元素一样小,则两个序列中有一个序列是清洁的')
        print('练习27.3-6 证明与0-1原则类似的关于双调排序网络的结论：一个能对任何0和1组成的双调序列进行排序的比较网络,也能对任何由任意数字组成双调序列进行排序')
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_4:
    '''
    chpater27.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.4 note

        Example
        ====
        ```python
        Chapter27_4().note()
        ```
        '''
        print('chapter27.4 note as follow')  
        print('27.4 合并网络')
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
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

class Chapter27_5:
    '''
    chpater27.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter27.5 note

        Example
        ====
        ```python
        Chapter27_5().note()
        ```
        '''
        print('chapter27.5 note as follow')  
        # python src/chapter27/chapter27note.py
        # python3 src/chapter27/chapter27note.py

chapter27_1 = Chapter27_1()
chapter27_2 = Chapter27_2()
chapter27_3 = Chapter27_3()
chapter27_4 = Chapter27_4()
chapter27_5 = Chapter27_5()

def printchapter27note():
    '''
    print chapter27 note.
    '''
    print('Run main : single chapter twenty-seven!')  
    chapter27_1.note()
    chapter27_2.note()
    chapter27_3.note()
    chapter27_4.note()
    chapter27_5.note()

# python src/chapter27/chapter27note.py
# python3 src/chapter27/chapter27note.py
if __name__ == '__main__':  
    printchapter27note()
else:
    pass
