# coding:utf-8
# usr/bin/python3
# python src/chapter34/chapter34note.py
# python3 src/chapter34/chapter34note.py
"""

Class Chapter34_1

Class Chapter34_2

Class Chapter34_3

Class Chapter34_4

Class Chapter34_5

"""
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    pass
else:
    pass

class Chapter34_1:
    """
    chapter34.1 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.1 note

        Example
        ====
        ```python
        Chapter34_1().note()
        ```
        """
        print('chapter34.1 note as follow')
        print('第34章 NP完全性')
        print('之前学习过的算法都是多项式时间算法:对规模为n的输入,它们在最坏情况下的运行时间为O(n^k),其中k为某个常数',
            '但是不是所有的问题都能在多项式时间内解决.例如图灵著名的“停机问题”,任何计算机不论耗费多少时间也不能解决')
        print('还有一些问题是可以解决的,但对任意常数k,都不能在O(n^k)的时间内得到解答.')
        print('一般来说,把可以由多项式时间的算法解决的问题看作是易处理的问题,而把需要超多项式时间才能解决的问题看作是难处理的问题')
        print('本章的主题是一类称为“NP完全”(NP-complete)的有趣问题,它的状态是未知的.直到现在,既没有人找出求解NP完全问题的多项式算法,',
            '也没有人能够证明对这类问题不存在多项式时间算法')
        print('这一所谓的P!=NP问题自1971年被提出之后,已经成为了理论计算机科学研究领域中,最深奥和最错综复杂的开放问题之一了')
        print('NP完全问题有一个方面特别诱人,就是在这一类问题中,有几个问题在表面上看来与有着多项式时间算法的问题非常相似',
            '在下面列出的每一对问题中,一个是可以在多项式时间内求解的,另一个是NP完全的,但从表面上看,两个问题之间差别很小')
        print('最短与最长简单路径：在第24章中,即使是在有负权值边的情况下,也能在一个有向图G=(V,E)中,在O(VE)的时间内,从一个源顶点开始找出最短的路径',
            '然而,寻找两个顶点间最长简单路径是NP完全的.事实上,即使所有边的权值都是1,它也是NP完全的')
        print('欧拉游程与哈密顿回路：对一个连通的有向图G=(V,E)的欧拉游程是一个回路,它遍历图G中每条边一次',
            '但可能不止一次地访问同一个顶点,可以在O(E)时间内,确定一个图中是否存在一个欧拉游程,并且能在O(E)时间内找出这一欧拉游程中的各条边',
            '一个有向图G=(V,E)的哈密顿回路是一种简单回路,它包含V中的每个顶点。确定一个有向图或者无向图中是否存在哈密顿回路是NP完全的')
        print('NP完全性与P类和NP类')
        print('  三类问题：P,NP和NPC,其中最后一类指NP完全问题.此处只是非形式地对它们进行描述,后面将给出更为形式的定义')
        print('  P类中包含的是在多项式时间内可解的问题.更具体一点,都是一些可以在O(n^k)时间内求解的问题,此处k为某个常数,n是问题的输入规模',
            '前面大多数算法绝大部分都属于P类')
        print('  NP类中包含的是在多项式时间内“可验证”的问题,此处是指某一解决方案的“证书”,就能够在问题输入规模的多项式时间内,验证该证书是正确的',
            '例如:在哈密顿回路问题中,给定一个有向图G=(V,E),证书可以是一个包含|V|个顶点的序列<v1,v2,v3,...,v|v|>',
            '很容易在多项式时间内判断出对i=1,2,3,...,|V|-1,是否有(vi,vi+1)∈E和(v|v|,v1)∈E')
        print('  P中的任何问题也都属于NP,这是因为如果某一问题是属于P的,则可以在不给出证书的情况下,在多项式时间内解决它.',
            '至于P是否是NP的一个真子集,在目前是一个开放的问题')
        print('  从非形式的意义上来说,如果一个问题属于NP,且与NP中的任何一个问题是一样“难的”(hard),写说它属于NPC类,也称它为NP完全的')
        print('  同时,不加证明地宣称如果任何NP完全的问题可以在多项式时间内解决,则每一个NP完全的问题都有一个多项式时间的算法')
        print('  从表面上看,很多自然而有趣的问题并不比排序,图的搜索或网络流问题更困难,但事实上,它们却是NP完全问题')
        print('NP完全性证明有点类似于8.1节中任何比较排序算法的运行时间下界Ω(nlgn)的证明;',
            '但是用于证明NP完全性所用到的特殊技巧却和8.1节中的决策树方法是不同的')
        print('证明一个问题为NP完全问题时,要依赖于三个关键概念：')
        print('  (1) 判定问题与最优化问题')
        print('  很多有趣的问题都是最优化问题(optimization problem),其中每一种可能的解都有一个相关的值,我们的目标是找出一个具有最佳值的可行解',
            '例如,在一个称为SHORTEST-PATH,已知的是一个无向图G及顶点u和v,要找出u到v之间的经过最少边的路径',
            '(换句话说,SHORTEST-PATH是在一个无权、无向图中的单点对间最短路径问题),',
            '然而,NP完全性不直接适合于最优化问题,但适合于判定问题(decision problem)',
            '因为这种问题的答案是简单的“是(1)”或“否(0)”')
        print('  通常,通过对待优化的值强加一个界,就可以将一个给定的最优化问题转换为一个相关的判定问题了')
        print('  例如,对SHORTEST-PATH问题来说,它有一个相关性的判定问题(称其为PATH),就是要判定给定的有向图G,顶点u和v,一个整数k,在u和v之间是否存在一条包含至多k条边的路径')
        print('  试图证明最优化问题是一个“困难的”问题时,就可以利用该问题与相关的判定问题之间的关系.',
            '这是因为,从某种意义上来说,判定问题要“更容易一些”,或至少“不会更难”')
        print('  例如,可以先解决SHORTEST-PATH问题,再将找出的最短路径上边的数目与相关判定问题中参数k进行比较,从而可以解决PATH问题',
            '换句话说,如果某个最优化问题比较容易的话,那么其相关的判定问题也会是比较容易的')
        print('  (2) 归约')
        print('  上述有关证明一个问题不难于或不简单于另一个问题的说法,对两个问题都是判定问题也是适用的',
            '在几乎每一个NP完全性证明中,都利用了这一思想.做法如下:考虑一个判定问题(称为A),希望在多项式时间内解决该问题.',
            '称某一特定问题的输入为该问题的一个实例,例如,PATH问题的一个实例可以是某一特定的图G、G中特定的点u和v和一个特定的整数k',
            '现在,假设有另一个不同的判定问题(称为问题B),知道如何在多项式时间内解决它.假设有一个过程,能将A的任何实例a转换成B的、具有以下特征的某个实例b:')
        print('  1) 转换操作需要多项式时间;')
        print('  2) 两个实例的答案是相同的.亦即,a的答案是“是”,当且仅当b的答案也是“是”')
        print('  称这样的一种过程为多项式时间的归约算法(reduction algorithm),并且提供了一种在多项式时间内解决问题A的方法:')
        print('  1) 给定问题A的一个实例a,利用多项式归约算法,将它转换为问题B的一个实例b')
        print('  2) 在实例b上,运行B的多项式时间判定算法')
        print('  3) 将b的答案用作a的答案')
        print('  只要上述步骤中的每一步只需多项式时间,则所有三步合起来也只需要多项式时间,这样就有了一种对a进行判断的方法')
        print('  换句话说,通过将对问题A的求解“归约”为对问题B的求解,就可以利用B的“易求解性”来证明A的“易求解性”')
        print('NP完全性是为了反映一个问题有多难,而不是为了反映它是多么容易.',
            '因此,以相反的方式来利用多项式时间归约,从而说明某一问题是NP完全的')
        print('至于NP完全性,不能假设问题A绝对没有多项式时间的算法,然而,证明的方法是类似的,即假设问题A是NP完全的前提下,来证明问题B是NP完全的')
        print('第一个NP完全问题')
        print('  应用归约技术要有一个前提,即已知一个NP完全的问题,这样才能证明另一个问题也是NP完全的.',
            '因此,需要找到一个“第一个NP完全问题”.将使用的这一个问题就是电路可满足性问题')
        print('  在这个问题中,已知的是一个布尔组合电路,它由AND、OR和NOT门组成,希望知道这个电路是否存在一组布尔输入,能够使它的输出为1')
        print('34.1 多项式时间')
        print('多项式时间可解问题的形式化定义')
        print('  (1) 把所需运行时间Θ(n^100)的问题作为难处理问题的合理之处,但在实际中,需要如此高次的多项式时间的问题是非常少的',
            '在实际中,所遇到的典型多项式时间可解问题所需的时间要少的多.经验表明：一旦某一问题的一个多项式时间算法被发现后,',
            '往往就会发现一些更为有效的算法,即使对某个问题来说,当前最佳算法的运行时间为Θ(n^100),很有可能在很短的时间内',
            '就能找到一个运行时间要好的多的算法')
        # !串行随机存取计算机模型
        print('  (2) 对很多合理的计算模型来说,在一个模型上用多项式时间可解的问题,在另一个模型上也可以在多项式时间内获得解决',
            '例如,在串行随机存取计算机模型上多项式可求解的问题类,与抽象的图灵机上在多项式时间内可求解的问题类是相同的,',
            '它也与利用并行计算机在多项式时间内可求解的问题类相同,即使处理器数目随输入规模以多项式增加也是这样')
        print('  (3) 多项式时间可解问题问题类具有很好的封闭性,这是因为在加法、乘法和组合运算下,多项式是封闭的.',
            '例如,如果一个多项式时间算法的输出馈送给另一个多项式时间算法作为输入,则得到的组合算法也是多项式时间的算法',
            '如果另外一个多项式时间算法对一个多项式时间的子程序进行常数次调用,那么组合算法的运行时间也是多项式的')
        print('抽象问题')
        print('  抽象问题Q为在问题实例集合I和问题解法集合S上的一个二元关系.例如,SHORTEST-PATH的一个实例是由一个图和两个顶点所组成的三元组,',
            '其解为图中的顶点序列,序列可能为空,表示两个顶点间不存在通路.问题SHORTEST-PATH本身就是一个关系,',
            '把图的每个实例和两个顶点与图中联系这两个顶点的最短路径联系在了一起.因为最短路径不一定是唯一的.',
            '因此,一个给定问题实例可能有多个解')
        print('编码')
        print('  如果要用一个计算机程序求解一个抽象问题,就必须用一种程序能理解的方式来表示问题实例.')
        print('  抽象对象集合S的编码是从S到二进制串集合的映射e.例如,都熟悉把自然数N={0,1,2,3,4,...}编码为{0,1,10,11,100,101,...}',
            '如bin(17)=10001')
        print(' ASCII编码、EBCDIC编码、UNICODE编码')
        print('多边形、图、函数、有序对、程序等所有这些都可以编码为二进制串')
        print('因此,“求解”某个抽象判定问题的计算机算法实际上把一个问题实例我的编码作为其输入.',
            '把实例集为二进制串的集合的问题称为具体问题.当提供给一个算法的是长度n=|i|的一个问题实例i时,算法可以在O(T(n))时间内产生问题的解,',
            '就说该算法在时间O(T(n))内解决了该具体问题。')
        print('因此，如果对某个常数k，存在一个算法能在时间O(n^k)内求解出某具体问题,就说该具体问题是多项式时间可解的')
        print('根据编码的不同,算法的运行时间可以是多项式时间或超多项式时间')
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
        print('')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_2:
    """
    chapter34.2 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.2 note

        Example
        ====
        ```python
        Chapter34_2().note()
        ```
        """
        print('chapter34.2 note as follow')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_3:
    """
    chapter34.3 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.3 note

        Example
        ====
        ```python
        Chapter34_3().note()
        ```
        """
        print('chapter34.3 note as follow')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_4:
    """
    chapter34.4 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.4 note

        Example
        ====
        ```python
        Chapter34_4().note()
        ```
        """
        print('chapter34.4 note as follow')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

class Chapter34_5:
    """
    chapter34.5 note and function
    """
    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter34.5 note

        Example
        ====
        ```python
        Chapter34_5().note()
        ```
        """
        print('chapter34.5 note as follow')
        # python src/chapter34/chapter34note.py
        # python3 src/chapter34/chapter34note.py

chapter34_1 = Chapter34_1()
chapter34_2 = Chapter34_2()
chapter34_3 = Chapter34_3()
chapter34_4 = Chapter34_4()
chapter34_5 = Chapter34_5()

def printchapter34note():
    """
    print chapter34 note.
    """
    print('Run main : single chapter thirty-four!')
    chapter34_1.note()
    chapter34_2.note()
    chapter34_3.note()
    chapter34_4.note()
    chapter34_5.note()

# python src/chapter33/chapter33note.py
# python3 src/chapter33/chapter33note.py

if __name__ == '__main__':  
    printchapter34note()
else:
    pass
