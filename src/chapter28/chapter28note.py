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
    chpater28.1 note and function
    """
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
        m = np.matrix([[1, 2, 3], [4, 5, 6]])
        print('一个2行3列的矩阵', m)
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
    '''
    chpater28.2 note and function
    '''
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
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_3:
    '''
    chpater28.3 note and function
    '''
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
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_4:
    '''
    chpater28.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter28.4 note

        Example
        ====
        ```python
        Chapter28_4().note()
        ```
        '''
        print('chapter28.4 note as follow')  
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

class Chapter28_5:
    '''
    chpater28.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter28.5 note

        Example
        ====
        ```python
        Chapter28_5().note()
        ```
        '''
        print('chapter28.5 note as follow')  
        # python src/chapter28/chapter28note.py
        # python3 src/chapter28/chapter28note.py

chapter28_1 = Chapter28_1()
chapter28_2 = Chapter28_2()
chapter28_3 = Chapter28_3()
chapter28_4 = Chapter28_4()
chapter28_5 = Chapter28_5()

def printchapter28note():
    '''
    print chapter28 note.
    '''
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
