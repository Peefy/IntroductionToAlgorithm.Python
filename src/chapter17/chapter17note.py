# coding:utf-8
# usr/bin/python3
# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
'''

Class Chapter17_1

Class Chapter17_2

Class Chapter17_3

Class Chpater17_4

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
from numpy import * 

class Chapter17_1:
    '''
    chpater17.1 note and function
    '''
    def multipop(self, S : list, k):
        '''
        栈的弹出多个数据的操作
        '''
        while len(S) > 0 and k > 0:
            S.pop()
            k = k - 1

    def increment(self, A : list):
        i = 0
        while i < len(A) and A[i] == 1:
            A[i] = 0
            i += 1
        if i < len(A):
            A[i] = 1

    def note(self):
        '''
        Summary
        ====
        Print chapter17.1 note

        Example
        ====
        ```python
        Chapter17_1().note()
        ```
        '''
        print('chapter17.1 note as follow')  
        # !在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的
        print('在平摊分析中，执行一系列数据结构的操作所需要的时间是通过对执行所有操作求平均而得出的')
        # !平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能
        print('平摊分析语平均情况分析的不同之处在于它不牵扯到概率；平摊分析保证在最坏情况下，每个操作具有平均性能')
        # !平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法
        print('平摊分析中三种最常用的技术：聚集分析，记账方法，势能方法')
        print('17.1 聚集分析')
        print('在聚集分析中，要证明对所有的n,由n个操作所构成的序列的总时间在最坏的情况下为T(n).')
        print('因此，在最坏情况下，每个操作的平均代价(或称平摊代价)为T(n)/n')
        print('这个平摊代价对每个操作都是成立的，即使当序列中存在几种类型的操作也是一样的')
        print('例1.栈操作')
        print(' 10.1介绍了两种基本的栈操作，每种操作的时间代价都是O(1)：PUSH和POP操作')
        print(' 因此，含n个PUSH和POP操作的序列的总代价为n,而这n个操作的实际运行时间就是Θ(n)')
        print(' 现在增加一个栈操作MULTIPOP(S,k),它的作用使弹出栈S的k个栈顶对象，或者当栈包含少于k个对象时，弹出整个栈中的数据对象')
        print(' 分析一个由n个PUSH，POP和MULTIPOP操作构成的序列，其作用于一个初始为空的栈。')
        print(' 序列中一次MULTIPOP操作的最坏情况代价为O(n),因为栈的大小至多为n')
        print(' 因此，任意栈操作的最坏情况就是O(n),因此n个操作的序列的代价是O(n^2),因为可能会有O(n)个MULTIPOP操作，每个的代价都是O(n)')
        print(' 利用聚集分析，可以获得一个考虑到n个操作的整个序列的更好的上界')
        print(' 对任意的n值，包含n个PUSH,POP和MULTIPOP操作的序列的总时间为O(n).每个操作的平均代价为O(n)/n=O(1)')
        print(' 把每个操作的平摊代价指派为平均代价。在这个例子中，三个栈操作的平摊代价都是O(1)')
        print('例2.二进制计数器递增1')
        print(' 作为聚集分析的另一个例子，考虑实现一个由0开始向上计数的k位二进制计数器的问题')
        print(' 使用一个位数组A[0..k-1]作为计数器。存储在计数器中的一个二进制数x的最低位在A[0]中，最高位在A[k-1]')
        print(' 每次INCREMENT操作的代价都与被改变值的位数成线性关系')
        print(' 如同栈的例子，大致的分析只能得到正确但不紧确的界')
        print(' 在最坏的情况下，INCREMENT的每次执行要花Θ(nk)')
        print(' 注意到在每次调用INCREMENT时，并不是所有的位都翻转，可以分析得更紧确一些')
        print(' 来得到n次INCREMENT操作的序列的最坏情况代价为O(n)')
        print(' 在每次调用INCREMENT时，A[0]确实都要发生翻转，下一个高位A[1]每隔一次翻转',
            '当作用于初始为零的计数器上时，n次INCREMENT操作会导致A[1]翻转[n/2]次')
        print(' 对于i>[lgn],位A[i]始终保持不变。在序列中发生的位翻转的总次数为2n')
        print(' 所以在最坏的情况下，作用于一个初始为零的计数器上的n次INCREMENTC操作的时间为O(n)。')
        print(' 每次操作的平均代价(即每次操作的平摊代价)是O(n)/n=O(1)')
        print(' 练习17.1-1 如果一组栈操作中包括了一次MULTIPUSH操作，它一次把k个元素压入栈内，',
            '那么栈操作的平摊代价的界O(1)还能够保持')
        print(' 练习17.1-2 证明：在k位计数器的例子中，如果包含一个DECREMENT操作，n个操作可能花费Θ(nk)的时间')
        print(' 练习17.1-3 最坏情况是这个数据结构的每个数据都是2的整数幂，运行时间为O(n),聚集分析的每次操作的平摊代价为O(n)/n=1')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_2:
    '''
    chpater17.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.2 note

        Example
        ====
        ```python
        Chapter17_2().note()
        ```
        '''
        print('chapter17.2 note as follow')
        print('17.2 记账方法')
        print('在平摊分析的记账方法中。对不同的操作赋予不同的费用，某些操作的费用比它们的实际代价或多或少')
        print('对一个操作的收费的数量称为平摊代价')
        print('当一个操作的平摊代价超过了它的实际代价时，两者的差值就被当做存款(credit),并赋予数据结构中的一些特定对象')
        print('存款可以在以后用于补偿那些平摊代价低于其实际代价的操作')
        # !就可以将一个操作的平摊代价看做两部分：其实际代价与存款
        print('就可以将一个操作的平摊代价看做两部分：其实际代价与存款')
        print('记账方法与聚集方法有着很大的不同，对后者而言，所有操作都具有相同的平摊代价')
        print('选择操作的平摊代价必须很小心。如果希望通过对平摊代价的分析来说明每次操作的最坏情况平均代价较小,',
            '则操作序列的总的平摊代价就必须是该序列的总的实际代价的一个上界')
        print('记账方法和聚集方法一样，这种关系必须对所有的操作序列都成立')
        print('例1.栈操作')
        print(' 对于栈操作例子的平摊分析的记账方法，各个栈操作的实际代价为')
        print('  PUSH 1;')
        print('  POP 1;')
        print('  MULTIPOP min(k, s)')
        print(' 其中k为MULTIPOP的一个参数,s为调用该操作时栈的大小。现在对它们赋值以下的平摊代价')
        print('  PUSH 2;')
        print('  POP 0;')
        print('  MULTIPOP 0;')
        print(' 注意MULTIPOP的平摊代价是常数(0),而它的实际代价却是个变量,此处所有三个平摊代价都是O(1)')
        print(' 但一般来说，从渐进的意义上看，所考虑的各种操作的平摊代价是会发生变化的')
        print('结论：对任意的包含n次PUSH，POP和MULTIPOP操作的序列，总的平摊代价就是其总的实际代价的一个上界')
        print(' 又因为总的平摊代价为O(n),故总的实际代价为O(n)')
        print('例2.二进制计数器递增1')
        # !二进制计数器递增1这个操作的运行时间与发生翻转的位数是成正比的
        print(' 位数在本例中将被用作代价。用1元钱表示单位代价(即某一位的翻转)')
        print(' 因为计数器中为1的位数始终是非负的，故存款的总额总是非负的')
        print('因此对n次INCREMENT操作，总的平摊代价为O(n),这就给出了总的实际代价的一个界')
        print('练习17.2-1 对一个大小始终不超过k的栈上执行一系列栈操作。在每k个操作后，复制整个栈的内容以留作备份')
        print(' 证明：在对各种栈操作赋予合适的平摊代价后，n个栈操作(包括复制栈的操作)的代价为O(n)')
        print('练习17.2-2 略')
        print('练习17.2-3 假设希望不仅能使一个计数器增值，也能使之复位至零',
            '如何将一个计数器实现为一个位数组，使得对一个初始为零的计数器，',
            '任一个包含n个INCREMENT和RESET操作的序列的时间为O(n)')
        print(' 可以保持一个指针指向高位1')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_3:
    '''
    chpater17.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.3 note

        Example
        ====
        ```python
        Chapter17_3().note()
        ```
        '''
        print('chapter17.3 note as follow')
        print('17.3 势能方法')
        # !在平摊分析中，势能方法不是将已预付的工作作为存储在数据结构特定对象中的存款来表示，
        # !而是表示成一种“势能”或“势”，在需要时可以释放出来，以支付后面的操作。势是与整个数据结构而不是其中的个别对象发生联系的
        print('在平摊分析中，势能方法不是将已预付的工作作为存储在数据结构特定对象中的存款来表示，')
        print('而是表示成一种“势能”或“势”，在需要时可以释放出来，以支付后面的操作。势是与整个数据结构而不是其中的个别对象发生联系的')
        print('不同的势函数可能会产生不同的平摊代价，但它们欧式实际代价的上界')
        print('在选择一个势函数时常要做一些权衡；可选用的最佳势函数的选择要取决于所需的时间界')
        print('势能方法的工作过程：开始时，先对一个初始数据结构D0执行n个操作')
        print('对每个i=1,2,...,n，设ci为第i个操作的实际代价，Di为对数据结构Di-1作用第i个操作的结果')
        print('每个操作的平摊代价为其实际代价加上由于该操作所增加的势')
        print('例1.栈操作')
        print('定义栈上的势函数Φ 为栈中对象的个数')
        print(' 三种栈操作中每一种的平摊代价都是O(1),这样包含n个操作的序列的总平摊代价就是O(n)')
        print(' 故n个操作的总平摊代价即为总的实际代价的一个上界。所以n个操作的最坏情况代价为O(n)')
        print('例2.二进制计数器递增1')
        print('定义第i次INCREMENT操作后计数器的势为bi，即第i次操作后计数器中1的个数')
        print('因为b0<=k,只要k=O(n),总的实际代价就是O(n)。如果执行了至少n=Ω(k)次INCREMENT操作')
        print('无论计数器中包含什么样的初始值，总的实际代价都是O(n)')
        print('练习17.3-1 假设有势函数Φ，使得对所有的i都有Φ(Di)>=Φ(D0),但是Φ(D0) ≠ 0.')
        print(' 证明：存在一个势函数Φ`，使得Φ`(D0) = 0, Φ`(D1) >= 0, 对所有i >= 1, 且用Φ`表示的平摊代价与用ΦB表示的平摊代价相同')
        print('练习17.3-2 用势能方法的分析重做练习17.1-3')
        print('练习17.3-3 考虑一个包含n个元素的普通儿茶最小堆数据结构')
        print('它支持最坏情况时间代价为O(lgn)的操作INSERT和EXTRACT-MIN.请给出一个势函数Φ，使得INSERT的平摊代价为O(lgn)')
        print('EXTRACT-MIN的平摊代价为O(1),并证明函数确实是有用的')
        print('练习17.3-4 假设某个栈在执行n个操作PUSH、POP和MULTIPOP之前包含s0个对象，结束后包含sn个对象')
        print('练习17.3-5 假设一个计数器的二进制表示中在开始时有b个1，而不是0。证明：如果n=Ω(b)')
        print(' 则执行n次INCREMENT操作的代价为O(n) (不能假设b是常数)')
        print('练习17.3-6 说明如何用两个普通的栈来实现一个队列，使得每个ENQUEUE和DEQUEUE操作的平摊代价都为O(1)')
        print('练习17.3-7 设计一个数据结构来支持整数集合S上的下列两个操作')
        print(' INSERT(S, x) 将x插入S中')
        print(' DELETE_LARGER_HALF(S) 删除S中最大的[S/2]个元素')
        print('解释如何实现这个数据结构，使得任意m个操作的序列在O(m)时间内运行')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

class Chapter17_4:
    '''
    chpater17.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter17.4 note

        Example
        ====
        ```python
        Chapter17_4().note()
        ```
        '''
        print('chapter17.4 note as follow')
        print('17.4 动态表')
        print('在某些应用中，无法预知要在表中存储多少个对象。为表分配了一定的空间，但后来发现空间并不够用')
        print('表的动态扩张和收缩问题。利用平摊分析，要证明插入和删除操作的平摊代价仅为O(1)')
        print('即使当它们引起了表的扩张和收缩时具有较大的实际代价也如此。')
        print('还可以看到如何来保证动态表中未用的空间始终不能超过整个空间的一个常量部分')
        print('使用在分析散列技术时的概念。定义一个非空表T的装载因子a(T)为表中存储的元素个数除以表的大小(槽的个数)后的结果')
        print('对一个空表(其中没有元素)定义其大小为0，其装载因子为1.')
        print('如果某一个动态表的装载银子以一个常数为上界，则表中未使用的空间就始终不会超过整个空间的一个常数部分')
        print('17.4.1 表扩张')
        print('假设一个表的存储空间分配为一个槽的数组。当所有的槽都被占用时，或者等价地，当其装载因子为1时，一个表就被填满了')
        print('当向一个满的表中插入一个项时，就能对原表进行扩张，即分配一个包含比原表更多槽的新表')
        print('因为总是需要这个表驻留在一段连续的内存中，必须为更大的表分配一个新的数组，然后把旧表中的项复制到新表中')
        print('一个常用的启发式方法是分配一个是原表两倍槽数的新表，如果只执行插入操作，则表的装载银子始终至少为1/2,浪费的空间就始终不会超过表空间的一半')
        print('这种启发式方法的特点是每当插入元素表被填满时，就重新分配两倍原来表空间的大小')
        print('如开始时表空间为1，当表满时，分配空间2，再满时分配空间4，以此类推，分配空间8，16，32')
        print('所以表扩张的程序的运行时间还由表中原来的元素个数决定')
        print('如果执行了n次操作，则依次操作的最坏情况代价为O(n),由此可得n次操作的总的运行时间的上界为O(n^2)')
        print('但是这个界不紧确，因为在执行n次插入操作的过程中，并不经常包括扩张表的代价')
        print('特别地，仅当i-1为2的整数幂时，第i次操作才会引起一次表的扩张')
        print('一次操作的平摊代价为O(1)')
        print('n次插入操作的总代价为3n,故每一次操作的平摊代价为3')
        print('用聚集方法，记账方法，势能方法分析')
        print('17.4.2 表扩张和收缩')
        print('为了实现TABLE_DELETE操作，只要将指定的项从表中去掉即可')
        print('但是当表的装载因子过小时，希望对表进行收缩，使得浪费的空间不致太大')
        print('表收缩与表扩张是类似的')
        print('当表中的项数降的过低时，就要分配一个新的、更小的表，而后将旧表的各项复制到新表中。')
        print('理想情况下，希望下面两个性质成立')
        print(' 动态表的装载因子由一个常数从下方限界')
        print(' 表操作的平摊代价由一个常数从上方限界')
        print('用基本插入和删除操作来度量代价')
        print('关于表收缩和扩张的一个自然的策略是当向满表中插入一个项时，将表的规模扩大一倍，',
            '而当从表中删除一个项就导致表不足半满')
        print('总之，因为每个操作的平摊代价都有一个常数上界，所以作用于一动态表上的n个操作的实际时间为O(n)')
        print('练习17.4-1 希望实现一个动态开放地址散列表。当表的装载因子达到某个严格小于1的数a时，就可以认为表满了')
        print(' 因为表的空间大小是一个有限的实数，并不是无限的(比如受内存的限制),所以当表还差一个元素未满时，装载因子此时严格小于1')
        print(' 说明如何对一个动态开放地址散列表进行插入，使得每个插入操作的平摊代价的期望值为O(1),',
            '为什么每个插入操作的实际代价的期望值不必是O(1)')
        print('练习17.4-2 证明：如果作用于一动态上的第i个操作是TABLE-DELETE,且装载因子大于0.5时')
        print(' 则以势函数表示的每个操作的平摊代价由一个常数从上方限界')
        print('练习17.4-3 假设当某个表的装载因子下降至1/4一下时，不是通过将其大小缩小一半来收缩')
        print(' 而是在表的装载因子低于1/3时，通过将其大小乘以2/3来进行收缩')
        print(' 利用势函数证明采用这种策略的TABLE-DELETE操作的平摊代价由一个常数从上方限界')
        print('思考题17-1 位反向的二进制计数器')
        # !FFT算法的第一步在输入数组A[0..n-1]上执行一次位反向置换
        print('FFT算法的第一步在输入数组A[0..n-1]上执行一次位反向置换')
        print('其中数组长度n=2^k(k为非负整数)。这个置换将某些元素进行交换，这些元素的下标的二进制表示彼此相反')
        print('revk(3)=12,revk(1)=8,revk(2)=4')
        print('比如n=16,k=4,写出一个能在O(nk)时间内完成对一个长度为n=2^k(k为非负整数)的数组的位反向置换算法')
        print('可以用一个基于平摊分析的算法来改善位反向置换的运行时间，方法是采用一个\"位反向计数器\"和一个程序',
            'BIT-REVERSED-INCREMENT该程序在给定一个位反向计数器和一个程序')
        print('如果k=4，且计数器的初始值位0，则连续调用BIT-REVERSED-INCREMENT就产生序列')
        print('0000,1000,0100,1100,0010,1010,...=0,8,4,12,2,10')
        print('假设可以在单位时间内对一个字左移或右移一位，可以实现一个O(n)时间的位反向置换')
        print('思考题17-2 使二叉查找动态化')
        # !对一个已经排序好的数组的二叉查找要花对数的时间
        print('对一个已经排序好的数组的二叉查找要花对数的时间,而插入一个新元素的时间则与数组的大小成线性关系')
        print('通过使用若干排好序的数组，就可以改善插入的时间')
        print('思考题17-3 平摊加权平衡树')
        print('假设在一棵普通的二叉查找树中，对每个结点x增加一个域size[x],',
            '表示在以x为根的子树中关键字的个数')
        print('设a是在范围1/2<=a<1之间的一个常数。')
        print('如果a * x.size >= x.left.size and a * x.size >= x.right.size',
            '就说一给定的结点x是a平衡的')
        print('如果树中每个结点都是a平衡的，则整棵树就是a平衡的')
        print('在最坏的情况下，对一棵有n个结点，a平衡的二叉查找树做一次查找要花O(lgn)时间')
        print('任何二叉查找树都具有非负的势；一棵1/2平衡树的势为0')
        print('对一棵包含n个结点的a平衡树，插入一个结点或删除一个结点需要O(lgn)平摊时间')
        print('思考题17-4 重构红黑树的代价')
        # !在红黑树上，有4种基本操作会做结构性的修改，它们是结点插入，结点删除、旋转以及颜色修改操作
        print('在红黑树上，有4种基本操作会做结构性的修改，它们是结点插入，结点删除、旋转以及颜色修改操作')
        print('描述一棵有n个结点的合法红黑树,使得调用RB-INSERT来插入第(n+1)个结点会引起Ω(lgn)次颜色修改')
        print('描述一棵有n个结点的合法红黑树，使得在一个特殊结点上调用RB-DELETE会引起Ω(lgn)次颜色修改')
        print('在最坏情况下，任何有m个RB-INSERT和RB-DELETE操作的序列执行O(m)次结构修改')
        # python src/chapter17/chapter17note.py
        # python3 src/chapter17/chapter17note.py

chapter17_1 = Chapter17_1()
chapter17_2 = Chapter17_2()
chapter17_3 = Chapter17_3()
chapter17_4 = Chapter17_4()

def printchapter17note():
    '''
    print chapter17 note.
    '''
    print('Run main : single chapter seventeen!')  
    chapter17_1.note()
    chapter17_2.note()
    chapter17_3.note()
    chapter17_4.note()

# python src/chapter17/chapter17note.py
# python3 src/chapter17/chapter17note.py
if __name__ == '__main__':  
    printchapter17note()
else:
    pass
