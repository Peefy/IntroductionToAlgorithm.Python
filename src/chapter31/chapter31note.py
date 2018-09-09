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
        print('欧几里得算法的推广形式')
        print('  现在来重写欧几里得算法以计算出其他的有用信息.特别地,推广该算法,使它能计算出满足下列条件的整系数x和y:d=gcd(a, b)=ax+by')
        print('  注意,x与y可能为0或负数.以后会发现这些系数对计算模乘法的逆是非常有用的,过程EXTENDED-EUCLID的输入为一对非负整数,返回一个满足上式的三元式(d,x,y)')
        print(_nt.euclid(30, 21))
        print('练习31.2-1 略')
        print('练习31.2-2 答案如下')
        print(_nt.extend_euclid(899, 493))
        print('练习31.2-3 证明对所有整数a, k, n; gcd(a, n) = gcd(a+kn, n)')
        print('练习31.2-4 仅用常量大小的存储空间(即仅存储常数个整数值)把过程EUCLID改写成迭代形式')
        print('练习31.2-5 如果a>b>=0,证明EUCLID(a, b)至多执行了1+logvb次递归调用.把这个界改进为1+logv(b / gcd(a, b))')
        print('练习31.2-6 过程EXTEND-EUCLID(Fk+1, Fk)返回1')
        print(_nt.extend_euclid(11, 8))
        print(_nt.extend_euclid(19, 11))
        print('练习31.2-7 用递归等式gcd(a0, a1,...,an)=gcd(a0, gcd(a1,...,an))来定义多余两个变量的gcd函数',
            '证明gcd函数的返回值与其自变量的次序无关,说明如何找出满足gcd(a0,a1,...,an)=a0x0+a1x1+..+anxn的整数x0,x1,...,xn',
            '证明所给出的算法执行的除法运算次数为O(n+lg(max{a0,a1,...,an}))')
        print('练习31.2-8 定义1cm(a0,a1,...,an)是n个整数a1,a2,...,an的最小公倍数,即每个ai的倍数中的最小非负整数',
            '说明如何用(两个自变量)gcd函数作为子程序以有效地计算出1cm(a1,a2,...,an)')
        print('练习31.2-9 证明n1,n2,n3和n4是两两互质的当且仅当gcd(n1n2,n3n4)=gcd(n1n3,n2n4)=1',
            '更一般地,证明n1,n2,...,nk是两两互质的,当且仅当从ni中导出的[lgk]对数互为质数')
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
        print('31.3 模运算')
        print('可以把模运算非正式地与通常的整数运算一样看待,如果执行模n运算,则每个结果值x都由集合{0,1,...,n-1}中的某个元素所取代',
            '该元素在模n的意义下与x等价(即用x mod n来取代x).如果仅限于运用加法、减法和乘法运算,则用这样的非正式模型就足够了.',
            '模运算模型最适合于用群论结构来进行描述')
        print('有限群')
        print('  群(S,+)是一个集合S和定义在S上的二进制运算+,它满足下列性质')
        print('  1) 封闭性：对所有a,b∈S,有a+b∈S,')
        print('  2) 单位元：存在一个一个元素e∈S,称为群的单位元,满足对所有a∈S,e+a=a+e=a')
        print('  3) 结合律：对所有a,b,c∈S,有(a+b)+c=a+(b+c)')
        print('  4) 逆元：对每个a∈S,存在唯一的元素b∈S,称为a的逆元,满足a+b=b+a=e')
        print('根据模加法与模乘法所定义的群')
        print('子群')
        print('  如果(S,+)是一个群,S‘∈S,并且(S’,+)也是一个群,则(S’,+)称为(S,+)的子群',
            '例如,在加法运算下,偶数形成一个整数的子群.下列定理提供了识别子群的一个有用的工具')
        print('定理31.14(一个有限群的非空封闭子集是一个子群) 如果(S,+)是一个有限群,S`是S的一个任意非空子集并满足：',
            '对所有a,b∈S’。则(S\',+)是(S,+)的一个子群')
        print('定理31.15(拉格朗日定理) 如果(S,+)是一个有限群,(S’,+)是(S,+)的一个子群,则|S‘|是|S|的一个约数')
        print('推论31.16如果S’是有限群S的真子群,则|S‘|<=|S|/2')
        print('由一个元素生成的子群')
        print('  定理31.17 对任意有限群(S,+)和任意a∈S,一个元素的阶等于它所生成的子群的规模,即ord(a)=|<a>|')
        print('  推论31.18 序列a(1),a(2),...是周期性序列,其周期为t=ord(a);即a(i)=a(j)当且仅当i=j(mod t)')
        print('  推论31.19 如果(S,+)是一个具有单位元e的有限群,则对所有a∈S')
        print('练习31.3-1 画出群(Z4,+4)和群(Z5*,*5)的运算表。通过找这两个群的元素间的、满足')
        print('练习31.3-2 证明定理31.14')
        print('练习31.3-3 证明：如果p是素数且e是正整数,则phi(e^e)=p^(e-1)(p - 1)')
        print('练习31.3-4 证明：对任意n>1和任意a∈Zn,由式fa(x)=ax mod n所定义的函数fa:Zn->Zn是Zn的一个置换')
        print('练习31.3-5 列举出Z9和Z13的所有子群')
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
        print('31.4 求解模线性方程')
        print('考虑求解下列方程的问题:ax=b(mod n)')
        print('其中a>0,n>0.这个问题有若干种应用;例如RSA公钥密码系统中,寻找密钥过程的一部分。假设已知a,b和n,希望求出所有满足式的对模n的x值',
            '可能没有解,也可能有一个或多个解')
        print('设<a>表示由a生成的Zn的子群.由于<a>={a(x):x>0}={ax mod n:x>0},所以方程有一个解当且仅当b∈<a>。',
            '拉格朗日定理告诉我们,|<a>|必定是n的约数,下列定义准确地刻画了<a>的特性')
        print('定理31.20 对任意整数a和n,如果d=gcd(a,n),则在Zn中')
        print('推论31.21 方程ax=b (mod n)对于未知量x有解,当且仅当gcd(a,n)|b')
        print('推论31.22 方程ax=b (mod n)或者对模n有d个不同的解,其中d=gcd(a, n),或者无解')
        print('定理31.23 设d=gcd(a,n),假定对整数x‘和y’,有d+ax‘+ny‘。如果d|b,则方程ax=b(mod b)有一个解的值为x0,满足',
            'x0=x’(b/d) mod n')
        print('定理31.24 假设方程ax=b (mod n)有解(即有d|b,其中d=gcd(a,n))x0是该方程的任意一个解,',
            '则该方程对模n恰有d个不同的解,分别为:xi=x0+i(n/d0)(i=1,2,...,d-1)')
        print('MODULAR-LINEAR-EQUATION-SOLVER执行O(lgn+gcd(a, n))次算术运算.因为EXTENDED-EUCLID需要执行O(lgn)次算术运算')
        print('推论31.25 对任意n>1,如果gcd(a, n)=1,则方程ax=b(mod n)对模n有唯一解')
        print('推论31.26 对任意n>1,如果gcd(a, n)=1,则方程ax=1(mod n)对模n有唯一解')
        print('练习31.4-1 求出方程35x=10(mod 50)的所有解')
        _nt.modular_linear_equation_solver(35, 10, 50)
        print('练习31.4-2 证明：当gcd(a, n)=1,由方程ax=ay(mod n)可得x=y(mod n).通过一个反例gcd(a, n)>1的情况来证明条件gcd(a,n)=1是必要的')
        print('练习31.4-3 考察下列对过程MODULAR-LINEAR-EQUATION-SOLVER的第3行修改')
        print('练习31.4-4 略')
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
        print('31.5 中国余数定理')
        print('大约在公元100年,中国的数学家孙子解决了一下问题:找出被3,5和7除时余数分别为2,3和2的所有整数x.有一个解为x=23;',
            '所有的解是形如23+105k(k为任意整数)的整数。\"中国余数定理\"提出,对一组两两互质的整数')
        print('中国余数定理:对一组两两互质的模数(如3,5和7)来说,其取模运算的方程组与对其积(如105)取模运算的方程之间存在着一种对应关系')
        print('中国余数定理有两个主要作用。设整数n因式分解为n=n1n2...nk,其中因子ni两两互质',
            '首先,中国余数定理是一个描述性的“结构定理”,说明Zn的结构等同于笛卡尔积Zn1×Zn2×...×Znk的结构')
        print('其中,第i个组元定义了对模ni的组元之间的加法与乘法运算',
            '其次,用这种描述常常可以获得有效的算法,因为处理Zn系统中的每个系统可能比处理模n运算效率更高(从位操作次数来看)')
        print('定理31.27(中国余数定理)设n=n1n2...nk,其中因子ni两两互质.考虑下列对应关系：a<->(a1,a2,...,ak)')
        print(' 其中a∈Zn,ai∈Zni,而且对i=1,2,...,k. ai = a mod ni')
        print('对Zn中元素所执行的运算可以等价地作用于对应的k元组,即在适当的系统中独立地对每个坐标位置执行所需的运算')
        print(' a<->(a1,a2,...,ak)')
        print(' b<->(b1,b2,...,bk)')
        print('推论31.28 如果n1,n2,...,nk两两互质,n=n1n2...nk,则对任意整数a1,a2,...,ak(i=1,2,...,k),方程组x=ai(mod ni)')
        print('推论31.29 如果n1,n2,...,nk两两互质,n=n1n2...nk,则对所有整数x和a(i=1,2,...,k)')
        print('练习31.5-1 计算出使方程x=4(mod 5)和x=5(mod 11)同时成立的所有解')
        print('练习31.5-2 试找出被9,8,7除时,余数分别为1,2,3的所有整数x')
        print('练习31.5-3 论证:在定理31.27的定义下,如果gcd(a, n)=1')
        print('练习31.5-4 证明:对于任意的多项式f,方程f(x)=0 (mod n)的根的数目等于每个方程',
            'f(x)=0 (mod n1),f(x)=0 (mod n2),...,f(x)=0(mod nk)的根的数目的积')
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
        print('31.6 元素的幂')
        print('正如考虑一个已知元素a对模n的倍数一样,常常自然地考虑对模n的a的幂组成的序列,其中a∈Zn:a0,a1,a2,a3,...')
        print('定理31.30 (欧拉定理) 对于任意整数n>1,a^phi(n)=1 (mod n)对所有a∈Zn都成立')
        print('定理31.31 (费马定理) 如果p是素数,则a^(p-1)=1 (mod p)对所有a∈Zp都成立')
        print('定理31.32 对所有素数p>2和所有正整数e,满足Zn为循环群的n(n > 1)值为2,4,p^e,2p^e')
        print('定理31.33 (离散对数定理) 如果g是Zn*的一个原根,则等式g^x=g^y(mod n)成立,当且仅当等式x=y(mod phi(n))成立')
        print('定理31.34 如果p是一个奇素数且e>=1,则方程x^2=1(mod p^e)')
        print('推论31.35 如果对模n存在1的非平凡平方根,则n是合数')
        print('运用反复平方法求数的幂')
        print('  数论计算中经常出现一种运算,就是求一个数的幂对另外一个数的模的运算,也称为模取幂',
            '更准确地说,希望找出一种有效的方法来计算a^b mod n的值,其中a,b为非负整数,n为正整数',
            '在许多素数测试子程序和RSA公开密钥加密系统中,模取幂运算是一种很重要的运算',
            '当用二进制来表示b时,采用反复平方法,可以有效地解决这个问题')
        print('  设<bk,bk-1,...,b1,b0>是b的二进制表示.(亦即,二进制表示有k+1位长,bk为最高有效位,b0为最低有效位))')
        print('  下列过程随着c的值从0到b成倍增长,最终计算出a^c mod n')
        print('练习31.6-1 试画出一张表以说明Z11中每个元素的阶.找出最小的原根g并计算出一张表,要求写出所有x∈Z11,相应的ind11.g(x)的值')
        print('练习31.6-2 写出一个模取值幂算法,要求该算法检查b的各位的顺序为从右向左,而不是从左向右')
        print('练习31.6-3 假设已知phi(n),试说明如何运用过程MODULAR-EXPONENTIATION计算出对任意a∈Zn,a^(-1) mod n')
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
        print('RSA公钥加密系统')
        print('公钥加密系统可以对传输于两个通信单位之间的消息进行加密,这样即使窃听者听到被加密的消息,也不能对加密消息进行破译',
            '公钥加密系统还能够使通信的一方,在电子报文的末尾附加一个无法伪造的“数字签名”')
        print('RSA公钥加密系统主要基于以下事实:寻求大素数是很容易的,但要把一个数分解为两个大素数的积确实相当困难的')
        print('公钥加密系统')
        print('  在公钥加密系统中,每个参与者都用一把公钥和一把密钥.没把密钥都是一条信息，例如在RSA公密钥加密系统中,每个密钥均是由一对整数组成')
        print('  在密码学中常把Alice和Bob作为例子：用P_A和S_A分别表示Alice的公钥和密钥;用P_B和S_B分别表示Bob的公钥和密钥')
        print('  每个参与者均自己创建起公钥和密钥。密钥需要保密，但公钥则可以对任何人公开或干脆公之于众。事实上，如果每个参与者的公钥都能在一个公开目录中查到的话是很方便的',
            '这样能使任何参与者容易地获得任何其他参与者的公钥')
        print('  公钥和密钥指定可适用于任何信息的功能函数.设D表示允许的信息集合。例如,D可能是所有有限长度的位序列集合')
        print('  在最简单的、原始的公钥密码学中，要求公钥与密钥说明一种从D到其自身的一一对应函数.对应于Alice的公钥P_A的函数用P_A()表示',
            '对应于她的密钥S_A的函数表示成S_A(),因此P_A()与S_A()函数都是D的排列.假定如果已知密钥P_A或S_A,就能够有效地计算出函数P_A()和S_A()')
        print('  任何参与者的公钥和密钥都是一个匹配对,他们制定的函数互为反函数,亦即对任何消息M∈D,有M=S_A(P_A(M));M=P_A(S_A(M))')
        print('在RSA公钥加密系统中,重要的是除Alice外,没有人能在较短时间内计算出S_A().送给Alice加密邮件的保密程度与Alice数组签名的真实性均依赖于以下假设:',
            '只有Alice能够计算出S_A().这个要求也是Alice要对S_A保密的原因：如果她不能做到这一点,就会失去她的唯一特性')
        print('  Bob取得Alice的公钥P_A')
        print('  Bod计算出响应于M的密文C=P_A(M),并把C发送给Alice')
        print('  当Alice收到密文C后,运用自己的密钥S_A恢复出原始信息：M=S_A(C)')
        print('类似地,在公钥系统中可以很容易地实现数字签名.假设现在Alice希望把一个数字签署的回应M发送给Bob.数')
        print('  Alice运用她的密钥S_A计算出信息M‘的数字签名o=S_A(M\')')
        print('  Alice把该消息/签名对(M\', o)发送给Bob')
        print('  当Bob收到(M\', o)时,他可以利用Alice的公钥通过验证等式M\'=P_A(o)来证实该消息的确是Alice发出的')
        print('因为数字签名有一条重要的性质,就是它可以被任何能取得签署者的公钥的人所验证.',
            '一条签署过的信息可以被一方确认过后再传送到其他地方,他们也同样能对该签名进行验证',
            '例如这条消息可能是Alice发给Bob的一张电子支票')
        print('RSA加密系统')
        print('  在RSA公钥加密系统中,一个参加者按下列过程来创建他的公钥与密钥')
        print('  1.随机选取两个大素数p和q,且p!=q,例如素数p和q可能各有512位(二进制)')
        print('  2.根据式n=pq计算出n的值')
        print('  3.选取一个与phi(n)互质的小奇数e,其中phi(n)=(p - 1)(q - 1)')
        print('  4.对模phi(n),计算出e的乘法逆元d的值')
        print('  5.输出对P=(e,n),把它作为RSA公钥')
        print('  6.把对S=(d,n)保密,并把它作为RSA密钥')
        print('运用MODULAR-EXPONENTIATION,来实现上述公钥与密钥的有关操作.为了分析这些操作的运行时间,假定公钥(e,n)和密钥(d,n)满足lge=O(1),',
            'lgd<=b以及lgn<=b,以及lgn<=b.则应用公钥操作,需要执行O(1)次模乘法运算和O(b^2)次位操作',
            '应用密钥操作,需要执行O(b)次模乘法运算和O(b^3)次位操作')
        print('使RSA与一个公开的单向散列函数h相结合,其中的单向三列函数是易于计算的,但是对这种函数来说,要找出两条消息M和M\'满足：h(M)和h(M\')',
            '这在计算上是不可行的.h(M)的值消息M的一个短的\"指纹\".')
        print('练习31.7-1 考察一个RSA密钥集合,其中p=11,q=29,n=319,e=3.在密钥中用到的d值应当是多少.对消息M=100加密后得到什么消息')
        print('练习31.7-2 证明：如果Alice的公开指数e等于3,并且对方获得了Alice的秘密指数d',
            '则对方能够在关于n的位数的多项式时间内对Alice的模n进行分解')
        print('练习31.7-3 证明：在如下意义中上说,RSA是乘法的：P_A(M1)P_A(M2)=P_A(M1M2)(mod n)')
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
        print('31.8 素数的测试')
        print('寻找大素数的问题，首先讨论素数的密度，接着讨论一种似乎可行的(但不完全)测试素数的方法,',
            '然后,介绍一种由Miller和Rabin发现的有效的随机素数测试算法')
        print('素数的密度')
        print('  在很多应用领域(如密码学中),需要找出大的“随机”素数,幸运的是,大素数并不算太少,因此测试适当的随机整数',
            '直至找到素数的过程也不是太费时的.素数分布函数pi(n)秒睡了小于或等于n的素数的数目,例如pi(10)=4',
            '因为小于或等于10的素数有4个,分别为2,3,5,7')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
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
