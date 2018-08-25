# coding:utf-8
# usr/bin/python3
# python src/chapter28/chapter28note.py
# python3 src/chapter28/chapter28note.py
"""

Class Chapter28_1

Class Chapter28_2

Class Chapter28_3

Class Chapter28_4

Class Chapter28_5

"""
from __future__ import absolute_import, division, print_function

import numpy as np

class Chapter28_1:
    """
    chapter28.1 note and function
    """

    def __init__(self):
        pass

    def note(self):
        """
        Summary
        ====
        Print chapter28.1 note

        Example
        ====
        ```python
        Chapter28_1().note()
        ```
        """
        print('chapter28.1 note as follow')
        print('28.1 矩阵的性质')
        print('矩阵运算在科学计算中非常重要')
        print('矩阵是数字的一个矩阵阵列,在python中使用np.matrix[[1,2],[3,4]]')
        print('矩阵和向量')
        print('单位矩阵')
        print('零矩阵')
        print('对角矩阵')
        print('三对角矩阵')
        print('上三角矩阵')
        print('下三角矩阵')
        print('置换矩阵')
        print('对称矩阵')
        print('矩阵乘法满足结合律,矩阵乘法对假发满足分配律')
        print('矩阵的F范数和2范数')
        print('向量的2范数')
        print('矩阵的逆,秩和行列式')
        print('定理28.1 一个方阵满秩当且仅当它为非奇异矩阵')
        print('定理28.2 当且仅当A无空向量,矩阵A列满秩')
        print('定理28.3 当且仅当A具有空向量时,方阵A是奇异的')
        print('定理28.4 (行列式的性质) 方阵A的行列式具有如下的性质')
        print('  如果A的任何行或者列的元素为0,则det(A)=0')
        print('  用常数l乘A的行列式任意一行(或任意一列)的各元素,等于用l乘A的行列式')
        print('  A的行列式的值与其转置矩阵A^T的行列式的值相等')
        print('  行列式的任意两行(或者两列)互换,则其值异号')
        print('定理28.5 当且仅当det(A)=0,一个n*n方阵A是奇异的')
        print('正定矩阵')
        print('定理28.6 对任意列满秩矩阵A,矩阵A\'A是正定的')
        print('练习28.1-1 证明：如果A和B是n*n对称矩阵,则A+B和A-B也是对称的')
        print('练习28.1-2 证明：(AB)\'=B\'A\',而且AA\'总是一个对称矩阵')
        print('练习28.1-3 证明：矩阵的逆是唯一的，即如果B和C都是A的逆矩阵,则B=C')
        print('练习28.1-4 证明：两个下三角矩阵的乘积仍然是一个下三角矩阵.',
            '证明：一个下三角(或者上三角矩阵)矩阵的行列式的值是其对角线上的元素之积',
            '证明：一个下三角矩阵如果存在逆矩阵,则逆矩阵也是一个下三角矩阵')
        print('练习28.1-5 证明：如果P是一个n*n置换矩阵,A是一个n*n矩阵,则可以把A的各行进行置换得到PA',
            '而把A的各列进行置换可得到AP。证明:两个置换矩阵的乘积仍然是一个置换矩阵',
            '证明：如果P是一个置换矩阵,则P是可逆矩阵,其逆矩阵是P^T,且P^T也是一个置换矩阵')
        print('练习28.1-6 设A和B是n*n矩阵，且有AB=I.证明:如果把A的第j行加到第i行而得到A‘',
            '则可以通过把B的第j列减去第i列而获得A’的逆矩阵B‘')
        print('练习28.1-7 设A是一个非奇异的n*n复数矩阵.证明：当且仅当A的每个元素都是实数时,',
            'A-1的每个元素都是实数')
        print('练习28.1-8 证明：如果A是一个n*n阶非奇异的对称矩阵,则A-1也是一个对称矩阵.',
            '证明：如果B是一个任意的m*n矩阵,则由乘积BAB^T给出的m*m矩阵是对称的')
        print('练习28.1-9 证明定理28.2。亦即，证明如果矩阵A为列满秩当且仅当若Ax=0,则说明x=0')
        print('练习28.1-10 证明:对任意两个相容矩阵A和B,rank(AB)<=min(rank(A),rank(B))',
            '其中等号仅当A或B是非奇异方阵时成立.(利用矩阵秩的另一种等价定义)')
        print('练习28.1-11 已知数x0,x1,...,xn-1,证明范德蒙德(Vandermonde)矩阵的行列式表达式')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_2:
    """
    chapter28.2 note and function
    """

    def __init__(self):
        pass

    def note(self):
        '''
        Summary
        ====
        Print chapter28.2 note

        Example
        ====
        ```python
        Chapter28_2().note()
        ```
        '''
        print('chapter28.2 note as follow')  
        print('28.2 矩阵乘法的Strassen算法')
        print('两个n*n矩阵乘积的著名的Strassen递归算法,其运行时间为Θ(n^lg7)=Θ(n^2.81)')
        print('对足够大的n,该算法在性能上超过了在25.1节中介绍的运行时间为Θ(n^3)的简易矩阵乘法算法MATRIX-MULTIPLY')
        print('算法概述')
        print('  Strassen算法可以看作是熟知的一种设计技巧--分治法的一种应用')
        print('  假设希望计算乘积C=AB,其中A、B和C都是n*n方阵.假定n是2的幂,把A、B和C都划分为四个n/2*n/2矩阵')
        print('  然后作分块矩阵乘法，可以得到递归式T(n)=8T(n/2)+Θ(n^2),但是T(n)=Θ(n^3)')
        print('Strassen发现了另外一种不同的递归方法,该方法只需要执行7次递归的n/2*n/2的矩阵乘法运算和Θ(n^2)次标量加法与减法运算')
        print('从而可以得到递归式T(n)=7T(n/2)+Θ(n^2),但是T(n)=Θ(n^2.81)')
        print('Strassen方法分为以下四个步骤')
        print(' 1) 把输入矩阵A和B划分为n/2*n/2的子矩阵')
        print(' 2) 运用Θ(n^2)次标量加法与减法运算,计算出14个n/2*n/2的矩阵A1,B1,A2,B2,...,A7,B7')
        print(' 3) 递归计算出7个矩阵的乘积Pi=AiBi,i=1,2,...,7')
        print(' 4) 仅使用Θ(n^2)次标量加法与减法运算,对Pi矩阵的各种组合进行求和或求差运算,',
            '从而获得结果矩阵C的四个子矩阵r,s,t,u')
        print('从实用的观点看,Strassen方法通常不是矩阵乘法所选择的方法')
        print('  1) 在Strassen算法的运行时间中,隐含的常数因子比简单的Θ(n^3)方法中的常数因子要大')
        print('  2) 当矩阵是稀疏的时候,为系数矩阵设计的方法更快')
        print('  3) Strassen算法不像简单方法那样具有数值稳定性')
        print('  4) 在递归层次中生成的子矩阵要消耗空间')
        # ! Strassen方法的关键就是对矩阵乘法作分治递归
        print('练习28.2-1 运用Strassen算法计算矩阵的乘积')
        print('矩阵的乘积为:')
        print(np.matrix([[1, 3], [5, 7]]) * np.matrix([[8, 4], [6, 2]]))
        print('练习28.2-2 如果n不是2的整数幂,应该如何修改Strassen算法,求出两个n*n矩阵的乘积',
            '证明修改后的算法的运行时间为Θ(n^lg7)')
        print('练习28.2-3 如果使用k次乘法(假定乘法不满足交换律)就能计算出两个3*3矩阵的乘积',
            '就能在o(n^lg7)时间内计算出两个n*n矩阵的乘积,满足上述条件的最大的k值是多少')
        print('练习28.2-4 V.Pan发现了一种使用132464次乘法的求68*68矩阵乘积的方法',
            '一种使用143640次乘法的求70*70矩阵乘积的方法',
            '一种使用155424次乘法的求72*72矩阵乘积的方法')
        print('练习28.2-5 用Strassen算法算法作为子程序,能在多长时间内计算出一个kn*n矩阵与一个n*kn矩阵的乘积')
        print('练习28.2-6 说明如何仅用三次实数乘法运算,就可以计复数a+bi与c+di的乘积.该算法应该把a,b,c和d作为输入,',
            '并分别生成实部ac-bd和虚部ad+bc的值') 
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_3:
    """
    chapter28.3 note and function
    """

    def __init__(self):
        pass

    def note(self):
        '''
        Summary
        ====
        Print chapter28.3 note

        Example
        ====
        ```python
        Chapter28_3().note()
        ```
        '''
        print('chapter28.3 note as follow')  
        print('28.3 求解线性方程组')
        print('对一组同时成立的线性方程组Ax=b求解时很多应用中都会出现的基本问题。一个线性系统可以表述为一个矩阵方程',
            '其中每个矩阵或者向量元素都属于一个域,如果实数域R')
        print('LUP分解求解线性方程组')
        print('LUP分解的思想就是找出三个n*n矩阵L,U和P,满足PA=LU')
        print('  其中L是一个单位下三角矩阵,U是一个上三角矩阵,P是一个置换矩阵')
        print('每一个非奇异矩阵A都有这样一种分解')
        print('对矩阵A进行LUP分解的优点是当相应矩阵为三角矩阵(如矩阵L和U),更容易求解线性系统')
        print('在计算出A的LUP分解后,就可以用如下方式对三角线性系统进行求解,也就获得了Ax=b的解')
        print('对Ax=b的两边同时乘以P,就得到等价的方程组PAx=Pb,得到LUx=Pb')
        print('正向替换与逆向替换')
        print('  如果已知L,P和b,用正向替换可以在Θ(n^2)的时间内求解下三角线性系统',
            '用一个数组pi[1..n]来表示置换P')
        print('LU分解的计算')
        print('  把执行LU分解的过程称为高斯消元法.先从其他方程中减去第一个方程的倍数',
            '以便把那些方程中的第一个变量消去')
        print('  继续上述过程,直至系统变为一个上三角矩阵形式,这个矩阵都是U.矩阵L是由使得变量被消去的行的乘数所组成')
        print('LUP分解的计算')
        print('  一般情况下,为了求线性方程组Ax=b的解,必须在A的非对角线元素中选主元以避免除数为0',
            '除数不仅不能为0,也不能很小(即使A是非奇异的),否则就会在计算中导致数值不稳定.因此,所选的主元必须是一个较大的值')
        print('  LUP分解的数学基础与LU分解相似。已知一个n*n非奇异矩阵A,并希望计算出一个置换矩阵P,一个单位下三角矩阵L和一个上三角矩阵U,并满足条件PA=LU')
        print('练习28.3-1 运用正向替换法求解下列方程组')
        print('练习28.3-2 求出下列矩阵的LU分解')
        print('练习28.3-3 运用LUP分解来求解下列方程组')
        print('练习28.3-4 试描述一个对角矩阵的LUP分解')
        print('练习28.3-5 试描述一个置换矩阵A的LUP分解,并证明它是唯一的')
        print('练习28.3-6 证明：对所有n>=1,存在具有LU分解的奇异的n*n矩阵')
        print('练习28.3-7 在LU-DECOMPOSITION中,当k=n时是否有必要执行最外层的for循环迭代?',
            '在LUP-DECOMPOSITION中的情况又是怎样?')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_4:
    """
    chapter28.4 note and function
    """
    def note(self):
        """
        Summary
        ====
        Print chapter28.4 note

        Example
        ====
        ```python
        Chapter28_4().note()
        ```
        """
        print('chapter28.4 note as follow')  
        print('28.4 矩阵求逆')
        print('在实际应用中,一般并不使用逆矩阵来求解线性方程组的解,而是运用一些更具数值稳定性的技术,如LUP分解求解线性方程组')
        print('但是,有时仍然需要计算一个矩阵的逆矩阵.可以利用LUP分解来计算逆矩阵')
        print('此外,还将证明矩阵乘法和计算逆矩阵问题是具有相同难度的两个问题,即(在技术条件限制下)可以使用一个算法在相同渐进时间内解决另外一个问题')
        print('可以使用Strassen矩阵乘法算法来求一个矩阵的逆')
        print('确实,正是由于要证明可以用比通常的办法更快的算法来求解线性方程组,才推动了最初的Strassen算法的产生')
        print('根据LUP分解计算逆矩阵')
        print('  假设有一个矩阵A的LUP分解,包括三个矩阵L,U,P,并满足PA=LU')
        print('  如果运用LU-SOLVE,则可以在Θ(n^2)的运行时间内,求出形如Ax=b的线性系统的解')
        print('  由于LUP分解仅取决于A而不取决于b,所以就能够再用Θ(n^2)的运行时间,求出形如Ax=b\'的另一个线性方程组的解')
        print('  一般地,一旦得到了A的LUP分解,就可以在Θ(kn^2)的运行时间内,求出k个形如Ax=b的线性方程组的解,这k个方程组只有b不相同')
        print('矩阵乘法与逆矩阵')
        print('  对矩阵乘法可以获得理论上的加速,可以相应地加速求逆矩阵的运算')
        print('  从下面的意义上说,求逆矩阵运算等价于矩阵乘法运算',
            '如果M(n)表示求两个n*n矩阵乘积所需要的时间,则有在O(M(n))时间内对一个n*n矩阵求逆的方法',
            '如果I(n)表示对一个非奇异的n*n矩阵求逆所需的时间,则有在O(I(n))时间内对两个n*n矩阵相乘的方法')
        print('定理28.7 (矩阵乘法不比求逆矩阵困难) 如果能在I(n)时间内求出一个n*n矩阵的逆矩阵',
            '其中I(n)=Ω(n^2)且满足正则条件I(3n)=O(I(n))时间内求出两个n*n矩阵的乘积')
        print('定理28.8 (求逆矩阵运算并不比矩阵乘法运算更困难) 如果能在M(n)的时间内计算出两个n*n实矩阵的乘积',
            '其中M(n)=Ω(n^2)且M(n)满足两个正则条件：对任意的0<=k<=n有M(n+k)=O(M(n)),以及对某个常数c<1/2有M(n/2)<=cM(n)',
            '则可以在O(M(n))时间内求出任何一个n*n非奇异实矩阵的逆矩阵')
        print('练习28.4-1 设M(n)是求n*n矩阵的乘积所需的时间,S(n)表示求n*n矩阵的平方所需时间',
            '证明:求矩阵乘积运算与求矩阵平方运算实质上难度相同：一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的矩阵平方算法,',
            '一个S(n)时间的矩阵平方算法蕴含着一个O(S(n))时间的矩阵相乘算法')
        print('练习28.4-2 设M(n)是求n*n矩阵乘积所需的时间,L(n)为计算一个n*n矩阵的LUP分解所需要的时间',
            '证明：求矩阵乘积运算与计算矩阵LUP分解实质上难度相同:一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的矩阵LUP分解算法',
            '一个L(n)时间的矩阵LUP分解算法蕴含着一个O(L(n))时间的矩阵相乘算法')
        print('练习28.4-3 设M(n)是求n*n矩阵的乘积所需的时间,D(n)表示求n*n矩阵的行列式的值所需要的时间',
            '证明：求矩阵乘积运算与求行列式的值实质上难度相同：一个M(n)时间的矩阵相乘算法蕴含着一个O(M(n))时间的行列式算法',
            '一个D(n)时间的行列式算法蕴含着一个O(D(n)时间的矩阵相乘算法')
        print('练习28.4-4 设M(n)是求n*n布尔矩阵的乘积所需的时间,T(n)为找出n*n布尔矩阵的传递闭包所需要的时间',
            '证明：一个M(n)时间的布尔矩阵相乘算法蕴含着一个O(M(n)lgn)时间的传递闭包算法,一个T(n)时间的传递闭包算法蕴含着一个O(T(n))时间的布尔矩阵相乘算法')
        print('练习28.4-5 当矩阵元素属于整数模2所构成的域时,基于定理28.8的求逆矩阵算法的是否能够运行？')
        print('练习28.4-6 推广基于定理28.8的求逆矩阵算法,使之能处理复矩阵的情形,并证明所给出的推广方法是正确的')
        print('   提示：用A的共轭转置矩阵A*来代替A的转置矩阵A^T,把A^T中的每个元素用其共轭复数代替就得到A*,也就是Hermitian转置')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_5:
    """
    chapter28.5 note and function
    """
    def note(self):
        """
        Summary
        ====
        Print chapter28.5 note

        Example
        ====
        ```python
        Chapter28_5().note()
        ```
        """
        print('chapter28.5 note as follow') 
        print('28.5 对称正定矩阵与最小二乘逼近') 
        print('对称正定矩阵有许多有趣而很理想的性质。例如，它们都是非奇异矩阵,并且可以对其进行LU分解而无需担心出现除数为0的情况')
        print('引理28.9 任意对称矩阵都是非奇异矩阵')
        print('引理28.10 如果A是一个对称正定矩阵,则A的每一个主子式都是对称正定的')
        print('设A是一个对称正定矩阵,Ak是A的k*k主子式,矩阵A关于Ak的Schur补定义为S=C-BAk^-1B^T')
        print('引理28.11 (Schur补定理) 如果A是一个对称正定矩阵,Ak是A的k*k主子式.则A关于Ak的Schur补也是对称正定的')
        print('推论28.12 对一个对称正定矩阵进行LU分解不会出现除数为0的情形')
        print('最小二乘逼近')
        print('对给定一组数据的点进行曲线拟合是对称正定矩阵的一个重要应用,假定给定m个数据点(x1,y1),(x2,y2),...,(xm,ym)',
            '其中已知yi受到测量误差的影响。希望找出一个函数F(x),满足对i=1,2,...,m,有yi=F(xi)+qi')
        print('其中近似误差qi是很小的,函数F(x)的形式依赖于所遇到的问题,在此,假定它的形式为线性加权和F(x)=∑cifi(x)')
        print('其中和项的个数和特定的基函数fi取决于对问题的了解,一种选择是fi(x)=x^j-1,这说明F(x)是一个x的n-1次多项式')
        print('这样一个高次函数F尽管容易处理数据,但也容易对数据产生干扰,并且一般在对未预见到的x预测其相应的y值时,其精确性也是很差的')
        print('为了使逼近误差最小,选定使误差向量q的范数最小,就得到一个最小二乘解')
        print('统计学中正态方程A^TAc=A^Ty')
        print('伪逆矩阵A+=(A^TA)^-1A^T')
        print('练习28.5-1 证明：对称正定矩阵的对角线上每一个元素都是正值')
        print('练习28.5-2 设A=[[a,b],[b,c]]是一个2*2对称正定矩阵,证明其行列式的值ac-b^2是正的')
        print('练习28.5-3 证明：一个对称正定矩阵中值最大的元素处于其对角线上')
        print('练习28.5-4 证明：一个对称正定矩阵的每一个主子式的行列式的值都是正的')
        print('练习28.5-5 设Ak表示对称正定矩阵A的第k个主子式。证明在LU分解中,det(Ak)/det(Ak-1)是第k个主元,为方便起见,设det(A0)=1')
        print('练习28.5-6 最小二乘法求')
        print('练习28.5-7 证明：伪逆矩阵A+满足下列四个等式：')
        print('  AA^+A=A')
        print('  A^+AA^+=A^+')
        print('  (AA^+)^T=AA^+')
        print('  (A^+A)^T=A^+A')
        print('思考题28-1 三对角线性方程组')
        print(' 1) 证明：对任意的n*n对称正定的三对角矩阵和任意n维向量b,通过进行LU分解可以在O(n)的时间内求出方程Ax=b的解',
            '论证在最坏情况下,从渐进意义上看,基于求出A^-1的任何方法都要花费更多的时间')
        print(' 2) 证明：对任意的n*n对称正定的三对角矩阵和任意n维向量b,通过进行LUP分解,'<
            '可以在O(n)的时间内求出方程Ax=b的解')
        print('思考题28-2 三次样条插值')
        print('  将一个曲线拟合为n个三次多项式组成')
        print('  用自然三次样条可以在O(n)时间内对一组n+1个点-值对进行插值')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

chapter28_1 = Chapter28_1()
chapter28_2 = Chapter28_2()
chapter28_3 = Chapter28_3()
chapter28_4 = Chapter28_4()
chapter28_5 = Chapter28_5()

def printchapter28note():
    """
    print chapter28 note.
    """
    print('Run main : single chapter twenty-eight!')
    chapter28_1.note()
    chapter28_2.note()
    chapter28_3.note()
    chapter28_4.note()
    chapter28_5.note()

# python src/chapter28/chapter28note.py
# python3 src/chapter28/chapter28note.py

if __name__ == '__main__':  
    printchapter28note()
else:
    pass
