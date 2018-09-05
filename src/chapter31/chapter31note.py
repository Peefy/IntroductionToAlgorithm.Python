# coding:utf-8
# usr/bin/python3
# python src/chapter31/chapter31note.py
# python3 src/chapter31/chapter31note.py
"""

Class Chapter31_1

Class Chapter31_2

Class Chapter31_3

Class Chapter31_4

Class Chapter31_5

Class Chapter31_6

Class Chapter31_7

Class Chapter31_8

Class Chapter31_9

"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np

if __name__ == '__main__':
    import numtheory as _nt
else:
    from . import numtheory as _nt

class Chapter31_1:
    """
    chapter31.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.1 note

        Example
        ====
        ```python
        Chapter31_1().note()
        ```
        """
        print('chapter31.1 note as follow')
        print('第31章 有关数论的算法')
        print('数论一度被认为是漂亮但是却没什么大用处的纯数学学科。有关数论的算法被广泛使用，部分是因为基于大素数的密码系统的范明')
        print('系统的安全性在于大素数的积难于分解，本章介绍一些初等数论知识和相关算法')
        print('31.1节介绍数论的基本概念，例如整除性、同模和唯一因子分解等')
        print('31.2节研究一个世界上很古老的算法；关于计算两个整数的最大公约数的欧几里得算法')
        print('31.3节回顾模运算的概念')
        print('31.4节讨论一个已知数a的倍数模n所得到的集合,并说明如何利用欧几里得算法求出方程ax=b(modn)的所有解')
        print('31.5节阐述中国余数定理')
        print('31.6节考察已知数a的幂模n所得的结果，并阐述一种已知a,b,n,可以有效计算a^b模n')
        print('31.7节描述RSA公开密钥加密系统')
        print('31.8节主要讨论随机性素数基本测试')
        print('31.9回顾一种把小整数分解因子的简单而有效的启发性方法,分解因子是人们可能想到的一个难于处理的问题',
            '这也许是因为RSA系统的安全性取决于对大整数进行因子分解的困难程度')
        print('输入的规模与算数运算的代价')
        print('  因为需要处理一些大整数,所以需要调整一下如何看待输入规模和基本算术运算的代价的看法',
            '一个“大的输入”意味着输入包含“大的整数”,而不是输入中包含“许多整数”(如排序的情况).',
            '因此,将根据表示输入数所要求的的位数来衡量输入的规模,而不是仅根据输入中包含的整数的个数',
            '具有整数输入a1,a2,...,ak的算法是多项式时间算法,仅当其运行时间表示lga1,lga2,...,lgak的多项式,',
            '即它是转换为二进制的输入长度的多项式')
        print('  发现把基本算术运算(乘法、除法或余数的计算)看作仅需一个单位时间的原语操作是很方便的',
            '但是衡量一个数论算法所需要的位操作的次数将是比较适宜的，在这种模型中，用普通的方法进行两个b位整数的乘法',
            '需要进行Θ(b^2)次位操作.')
        print('一个b位整数除以一个短整数的运算,或者求一个b位整数除以一个短整数所得的余数的运算,也可以用简单算法在Θ(b^2)的时间内完成',
            '目前也有更快的算法.例如,关于两个b位整数相乘这一运算,一种简单分治算法的运行时间为Θ(b^lg2(3))',
            '目前已知的最快算法的运行时间为Θ(blgblglgb),在实际应用中,Θ(b^2)的算法常常是最好的算法,将用这个界作为分析的基础')
        print('在本章中,在分析算法时一般既考虑算术运算的次数,也考虑它们所要求的位操作的次数')
        print('31.1 初等数论概念')
        print('整数性和约数')
        print('  一个整数能被另一个整数整除的概念是数论中的一个中心概念。记号d|a(d整数a),意味着对某个整数k,有a=kd',
            '0可被任何整数整除.如果a>0且d|a,则|d|<=|a|.如果d|a,则也可以说a是d的倍数。')
        print('素数和合数')
        print('  对于某个整数a>1，如果它仅有平凡约数1和a,则称a为素数(或质数)。素数具有许多特殊性质,在数论中起着关键作用.按顺序看,前20个素数',
            '2,3,5,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71')
        print('  证明：有无穷多个素数。不是素数的整数a>1称为合数。例如，因为有3|39,所以39是合数。整数1被称为基数，他既不是素数也不是合数',
            '类似地,整数0和所有负整数既不是素数，也不是合数')
        print('除法定理，余数和同模')
        print('  已知一个整数n,所有整数都可以划分是n的倍数的整数,以及不是n的倍数的整数。对于不是n的倍数的那些整数',
            '又可以根据它们除以n所得的余数来进行分类,数论的大部分理论都是基于上述划分的')
        print('定理31.1(除法定理) 对任意整数a和任意正整数n,存在唯一的整数q和r,满足0<=r<n,并且a=qn+r',
            '值q=[a/n]称为除法的商.值r=a mod n称为除法的余数。n|a当且仅当a mod n=0')
        print('根据整数模n所得的余数,可以把整数分成n个等价类.包含整数a的模n等价类为：')
        print('  [a]n={a+kn : k∈Z}')
        print(' 如果用0表示[0]n,用1表示[1]n等等,每一类均用其最小的非负元素来表示,',
            '提到Zn的元素-1就是指[n-1]n,因为-1=n-1(mod n)')
        print('公约数与最大公约数')
        print('  如果d是a的约束并且也是b的约数,则d是a与b的公约数.例如30的约数为1,2,3,5,6,10,15,30,因此24与30的公约数为1,2,3和6',
            '注意，1是任意两个整数的公约数')
        print('  公约数的一条重要性质为：d|a并且d|b蕴含着d|(a+b)并且d|(a-b)')
        print('  更一般地，对任意整数x和y,有d|a并且d|b蕴含着d|(ax+by);同样,如果a|b,则或者|a|<=|b|,或者b=0,这说明:a|b且b|a蕴含着a=±b',
            '两个不同时为0的整数a与b的最大公约数表示成gcd(a,b),例如gcd(24,30)=6,gcd(5,7)=1,gcd(0,9)=9',
            '如果a与b不同时为0,则gcd(a,b)是一个在1与min(|a|,|b|)之间的整数,定义gcd(0,0)=0')
        print('下列性质是gcd函数的基本性质:')
        print('  gcd(a,b)=gcd(b,a)')
        print('  gcd(a,b)=gcd(-a,b)')
        print('  gcd(a,b)=gcd(|a|,|b|)')
        print('  gcd(a,0)=|a|')
        print('  gcd(a,ka)=|a|,对任何k∈Z')
        print('定理31.2 如果a和b是不都为0的任意整数,则gcd(a,b)是a与b的线性组合集合{ax+by:x,z∈Z}中的最小正元素')
        print('推论31.3 对任意整数a与b,如果d|a并且d|b,则d|gcd(a, b)')
        print('推论31.4 对所有整数a和b以及任意非负整数n,gcd(an,bn)=ngcd(a,b)')
        print('互质数')
        print('  如果两个整数a与b仅有公因数1,即如果gcd(a,b)=1,则a与b称为互质数,则它们的积与p互为质数')
        print('定理31.6 对任意整数a,b和p,如果gcd(a, p)=1且gcd(b,p)=1,则gcd(ab,p)=1')
        print('唯一的因子分解')
        print('定理31.7 对所有素数p和所有整数a,b,如果p|ab,则p|a或p|b(或两者都成立)')
        print('定理31.8 (唯一质因子分解) 合数a仅能以一种方式,写成如下的乘积形式')
        print('练习31.1-1 证明有无穷多个素数。(提示：证明素数p1,p2,...,pk都不能整除(p1p2...pk)+1)')
        print('练习31.1-2 证明：如果a|b且b|c,则a|c')
        print('练习31.1-3 证明：如果p是素数并且0<k<p,则gcd(k, p)=1')
        print('练习31.1-4 证明推论31.5')
        print('练习31.1-5 证明：如果p是素数且0<k<p.证明对所有整数a,b和素数p')
        print('练习31.1-6 证明：如果a和b是任意整数,且满足a|b和b>0,则对任意x有：(x mod b) mod a == x mod a')
        print('练习31.1-7 对任意整数k>0,如果存在一个整数a满足a^k=n,则说整数n为k次幂',
            '如果对于某个整数k>1,n>1是一个k次幂,则说n是非平凡幂.说明如何关于b的多项式时间内,确定出一个b位整数n是非平凡幂')
        print('练习31.1-8 略')
        print('练习31.1-9 证明：gcd运算满足结合律。亦即证明对所有整数a,b,c,有gcd(a,gcd(b,c))=gcd(gcd(a,b),c)')
        print('练习31.1-10 证明定理31.8 ')
        print('练习31.1-11 试写出一个b位整数除以一个短整数的有效算法,以及求一个n位整数除以一个短整数的余数的有效算法.所给出的算法的运行时间应为O(b^2)')
        print('练习31.1-12 写出一个能把一个b位二进制数转化为相应的十进制表示的有效算法.论证：如果长度至多为b的整数的乘法与除法运算所需时间为M(b),',
            '则执行二进制到十进制转换所需的时间为Θ(M(b)lgb).(运用分治法,把数分为顶部和底部两部分,分别进行递归操作而获得所需结果)')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_2:
    """
    chapter31.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.2 note

        Example
        ====
        ```python
        Chapter31_2().note()
        ```
        """
        print('chapter31.2 note as follow')
        print('31.2 最大公约数')
        print('运用欧几里得算法有效地计算出两个整数的最大公约数,在对其运行时间分析中,会发现与斐波那契数存在着联系,由此可获得欧几里得算法在最坏情况下的输入')
        print('在本节中仅限于对非负整数的情况讨论gcd(a, b) = gcd(|a|, |b|)')
        print('原则上讲,可以根据a和b的素数因子分解,求出正整数a和b的最大公约数gcd(a,b)')
        print('如果a=p1^e1 p2^e2 p3^e3...pr^er; a=p1^f1 p2^f2 p3^f3...pr^fr;')
        print('其中使用了零指数,使得素数集合p1,p2,...,pr对于a和b相同')
        print('  gcd(a,b)=p1^min(e1,f1) p2^min(e2,f2) ...pr^min(er,fr)')
        print('目前已知的最好的分解因子算法也不能达到多项式的运行时间,因此,根据这种方法来计算最大公约数,不大可能获得一种有效的算法')
        print('定理31.9 (GCD递归定理) 对任意负整数a和任意的正整数b')
        print(' gcd(a,b)=gcd(b, a mod b)')
        print('欧几里得算法')
        print(_nt.euclid(30, 21))
        print('欧几里得算法的运行时间')
        print('  最坏情况下,可以把欧几里得算法看成输入a与b的大小的函数。不失一般性,假定a>b>=0',
            '这个假设的合理性是基于下述观察的:如果b>a>=0,则EUCLID(a,b)立即会递归调用EUCLID(b, a),即如果第一个自变量小于第二个自变量',
            '则EUCLID进行一次递归调用以使两个自变量兑换,然后继续往下执行.类似地,如果b=a>0,则过程在进行一次递归调用后就终止执行,因为a mod b=0')
        print('引理31.10 如果a>b>=1并且EUCLID(a, b)执行了k>=1次递归调用,则a>=Fk+2,b>=Fk+1')
        print('定理31.11 (Lame定理) 对任意整数k>=1,如果a>b>=1且b<Fk+1,则EUCLID(a, b)的递归调用次数少于k次')
        print('由于Fk约为v^k/sqrt(5),其中v是定义的黄金分割率(1+sqrt(5))/2,所以EUCLID执行中的递归调用次数O(lgb)',
            '如果过程EUCLID作用于两个b位数,则它执行O(b)次算术和O(b^3)次位操作')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_3:
    """
    chapter31.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.3 note

        Example
        ====
        ```python
        Chapter31_3().note()
        ```
        """
        print('chapter31.3 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_4:
    """
    chapter31.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.4 note

        Example
        ====
        ```python
        Chapter31_4().note()
        ```
        """
        print('chapter31.4 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_5:
    """
    chapter31.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.5 note

        Example
        ====
        ```python
        Chapter31_5().note()
        ```
        """
        print('chapter31.5 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_6:
    """
    chapter31.6 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.6 note

        Example
        ====
        ```python
        Chapter31_6().note()
        ```
        """
        print('chapter31.6 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_7:
    """
    chapter31.7 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.7 note

        Example
        ====
        ```python
        Chapter31_7().note()
        ```
        """
        print('chapter31.7 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_8:
    """
    chapter31.8 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.8 note

        Example
        ====
        ```python
        Chapter31_8().note()
        ```
        """
        print('chapter31.8 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

class Chapter31_9:
    """
    chapter31.9 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter31.9 note

        Example
        ====
        ```python
        Chapter31_9().note()
        ```
        """
        print('chapter31.9 note as follow')
        # python src/chapter31/chapter31note.py
        # python3 src/chapter31/chapter31note.py

chapter31_1 = Chapter31_1()
chapter31_2 = Chapter31_2()
chapter31_3 = Chapter31_3()
chapter31_4 = Chapter31_4()
chapter31_5 = Chapter31_5()
chapter31_6 = Chapter31_6()
chapter31_7 = Chapter31_7()
chapter31_8 = Chapter31_8()
chapter31_9 = Chapter31_9()

def printchapter31note():
    """
    print chapter31 note.
    """
    print('Run main : single chapter thirty-one!')
    chapter31_1.note()
    chapter31_2.note()
    chapter31_3.note()
    chapter31_4.note()
    chapter31_5.note()
    chapter31_6.note()
    chapter31_7.note()
    chapter31_8.note()
    chapter31_9.note()

# python src/chapter31/chapter31note.py
# python3 src/chapter31/chapter31note.py

if __name__ == '__main__':  
    printchapter31note()
else:
    pass
