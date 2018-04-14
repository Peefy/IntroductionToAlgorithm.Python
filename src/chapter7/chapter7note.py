# python src/chapter7/chapter7note.py
# python3 src/chapter7/chapter7note.py
'''
Class Chapter7_1

Class Chapter7_2

Class Chapter7_3

Class Chapter7_4
'''

from __future__ import division, absolute_import, print_function

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange

if __name__ == '__main__':
    import quicksort
    import stooge
else:
    from . import quicksort
    from . import stooge

class Chapter7_1:
    def note(self):
        '''
        Summary
        ====
        Print chapter7.1 note

        Example
        ====
        >>> Chapter7_1().note()
        '''
        print('chapter7.1 note as follow')
        print('第7章 快速排序')
        print('快速排序是一种排序算法，对包含n个数的输入数组进行排序，最坏情况的运行时间为Θ(n^2)')
        print('虽然这个最坏情况运行时间比较差，但是快速排序通常是用于排序最佳的实用选择，这是因为其平均性能相当好')
        print('快速排序期望的运行时间为Θ(nlgn),且Θ(nlgn)记号中隐含的常数因子很小')
        print('快速排序能够进行就地排序，在虚存坏境中也能很好地工作')
        print('7.1 快速排序的描述')
        print('像合并排序一样，快速排序也是基于分治模式的')
        print(' 1.分解:数组A[p..r]被划分成两个(可能为空的)子数组A[p..q-1]和A[q+1..r]')
        print('  使得A[p..q-1]中的每个元素都小于等于A(q),而且，小于等于A[q+1..r]')
        print('  下标q也在这个划分过程中进行计算')
        print(' 2.解决:通过递归调用快速排序，对子数组A[p..q-1]和A[q+1..r]排序')
        print(' 3.合并:因为这两个子数组是就地排序的(不开辟新的数组),将他们合并不需要任何操作，整个数组A[p..r]已经排好序')
        print('子数组快速排序伪代码')
        print('QUICKSORT(A,p,r)')
        print(' 1. if q < r')
        print(' 2.   q <- PARTITION(A,p,r)')
        print(' 3.       QUICKSORT(A,p,q-1)')
        print(' 3.       QUICKSORT(A,q+1,r)')
        print('排序完整的数组A，调用QUICKSORT(A,0,len(A))即可')
        print('快速排序算法的关键是PARTITION过程，它对子数组A[q..r]进行就地重排')
        print('PARTITION(A,p,r)')
        print(' 1. x <- A[r]')
        print(' 2. i <- p-1')
        print(' 3. for j <- p to r-1')
        print(' 4.  if A[j] <= x')
        print(' 5.      i <- i+1')
        print(' 6.      exchange A[i] <-> A[j]')
        print(' 7. exchange A[i+1] <-> A[r]')
        print(' 8. return i + 1')
        A = [8, 9, 6, 7, 4, 5, 2, 3, 1]
        print('数组A', _deepcopy(A), '的快速排序过程为:', 
            quicksort.quicksort(A))
        A = [13, 19, 9, 5, 12, 8, 7, 4, 11, 2, 6, 21]
        print('练习7.1-1 数组A', _deepcopy(A), 
            '的一步partition过程得到middle索引为：', 
            quicksort.partition(A, 0, len(A) - 1))
        A = [11, 11, 11, 11, 11]
        print('练习7.1-2 数组A', _deepcopy(A), 
            '的一步partition过程得到middle索引为：', 
            quicksort.partition(A, 0, len(A) - 1))
        print('练习7.1-3 就一个长度为n的for循环，且一定会执行，所以时间复杂度为Θ(n)，然后用确界的夹逼定义证明')
        print('练习7.1-4 不等号方向改变即可')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_2:
    def note(self):
        '''
        Summary
        ====
        Print chapter7.2 note

        Example
        ====
        >>> Chapter7_2().note()
        '''
        print('chapter7.2 note as follow')
        print('7.2 快速排序的性能')
        print('快速排序的运行时间与划分是否对称有关，而后者由与选择了哪个元素来进行划分有关')
        print('如果划分是对称的，那么快速排序算法从渐进上与合并算法一样快，否则就和插入排序一样慢')
        print('快速情况的最坏情况划分行为发生在划分过程产生的两个区域分别包含n-1个元素和1个0元素的时候')
        print('假设每次划分都出现了这种不对称划分，划分的时间代价为Θ(n),故算法的运行时间可以递归地写为')
        print('T(n)=T(n-1)+T(0)+Θ(n),递归式的解为T(n)=Θ(n^2)')
        print('快速排序的最坏情况并不比插入排序的最坏情况更好')
        print('另外，当一个已经排序好时，快速排序运行时间Θ(n^2)，插入排序运行时间Θ(n)')
        print('快速排序最佳情况是是其中一个字问题的大小为[n/2],另一个问题的大小为[n/2]-1')
        print('在这种情况下，快速排序的运行时间要快的多，T(n)<=T(n/2)+Θ(n)')
        print('根据主定理，以上递归式的解为O(nlgn)')
        print('平衡的划分')
        print('快速排序的平均运行时间与其最佳运行时间很接近')
        print('练习7.2-1 T(1)=T(0)+Θ(n),T(2)=T(1)+Θ(n),T(n)=T(n-1)+Θ(n)')
        print(' 从第一个式子加到第n个式子，T(n)=Θ(n^2)')
        print('练习7.2-2 数组A中的每个元素都相同时，也属于元素已经排序好的情况，',
            '所以调用PARTITION子程序每次都会得到最差的分配，所以最坏情况运行时间为T(n)=Θ(n^2)')
        print('练习7.2-3 根据书中的描写降序排序好的元素，会导致每次分配都得到最差的情况，又递归式和主定理得T(n)=Θ(n^2)')
        print('练习7.2-4 对已经排序好的支票对于快速排序来说属于最差情况输入，运行时间为O(n^2)')
        print(' 而对于插入排序来说却是最优输入，运行时间为O(n)')
        print('练习7.2-5 [滑稽]但是平均的运行时间仍然为O(nlgn)')
        print('练习7.2-6 略')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_3:
    '''
    chapter7.3 content : note, function, etc..

    See Also
    ========
    Chapter7_1 Chapter7_2 Chapter7_4
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter7.3 note

        Example
        ====
        >>> Chapter7_3().note()
        '''
        print('chapter7.3 note as follow')
        print('7.3 快速排序的随机化版本')
        print('在探讨快速排序的平均性态过程中，假定输入数据的所有排列都是等可能的')
        print('但在工程中，这个假设就不会总是成立')
        print('虽然第五章介绍过一些随机算法，但是如果采用一种不同的，称为随机取样的随机化技术的话，可以使分析更加简单')
        print('在这种方法中，不是时钟采用A[r]作为主元，而是从子数组A[p..r]中随机选择一个元素')
        print('然后将这个随机元素与A[r]交换作为主元')
        print('因为主元元素是随机选择的，在期望的平均情况下，对输入数组的划分比较对称')
        A = [8, 7, 6, 5, 4, 3, 2, 1]    
        print('数组[8, 7, 6, 5, 4, 3, 2, 1]的随机化快速排序：', 
            quicksort.randomized_quicksort(A))
        print('练习7.3-1:大部分时候输入的待排序序列我们是不知道的，而对于快速排序来讲，一个平均的输入才能反映其算法性能，最坏情况出现的概率比较小')
        print('练习7.3-2:最佳情况调用Θ(n)次，最坏情况调用Θ(n^2)次')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

class Chapter7_4:
    '''
    chapter7.4 content : note, function, etc..

    See Also
    ========
    Chapter7_1 Chapter7_2 Chapter7_3
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter7.4 note

        Example
        ====
        ``Chapter7_4().note()``
        '''
        print('chapter7.4 note as follow')
        print('7.4 快速排序分析')
        print('7.4.1 最坏情况分析')
        print('如果快速排序中每一层递归上所做的都是最坏情况划分，则运行时间为Θ(n^2)')
        print('7.4.2 期望的运行时间')
        print('RANDOMZIED-QUICKSORT的平均情况运行时间为O(nlgn)')
        print('运行时间和比较')
        print('quicksort的运行时间是由花在过程PARTITION上的时间所决定的。')
        print('每当PARTITION过程被调用时，就要选出一个主元元素，后续对QUICKSORT和PARTITION的各次递归调用中，都不会包含该元素')
        print('于是，在快速排序算法的整个执行过程中，最多只可能调用PARTITION过程n次，调用一次PARTITION的时间为O(1)在加上一段时间')
        print('引理7.1 设当QUICKSORT在一个包含n个元素的数组上运行时，PARTITION在第四行所做的比较次数为X,那么QUICKSORT的运行时间为O(n+X)')
        print('练习7.4-1 递归式子T(n)=max(T(q)+T(n-q-1)+Θ(n))中，T(n)=Ω(n^2)')
        print('练习7.4-2 快速排序的最佳情况运行时间为Ω(nlgn)')
        print('练习7.4-3 略')
        print('练习7.4-4 RANDOMIZED-QUICKSORT算法期望的运行时间为Ω(nlgn)')
        print('练习7.4-5 对插入排序来说，当其输入已经是几乎排好序的，运行时间是很快的')
        print(' 当在一个长度小于k的子数组上调用快速排序时，让它不做任何排序就返回。', 
            '当顶层的快速排序调用返回后，对整个数组运行插入排序来完成排序过程。', 
            '这一排序算法的期望运行时间为O(nk+nlg(n/k))')
        print('练习7.4-6 PARTITION过程做这样的修改，从数组A中随机地选出三个元素，并围绕这三个数的中数(即这三个元素的中间值)进行划分', 
            '求出以a的函数形式表示的、最坏情况中a:(1-a)划分的近似概率')
        A = [13, 19, 9, 5, 12, 8, 7, 4, 11, 2, 6, 21]
        print('思考题7-1：数组A', _deepcopy(A), '的HOARE-PARTITION算法过程为:', 
            quicksort.hoare_partition(A, 0, len(A) - 1))
        print('数组A', _deepcopy(A), '的HOARE-QUICKSORT的过程为：', quicksort.hoare_quicksort(A))
        print('思考题7-2:对快速排序算法的另一种分析')
        print(' 着重关注每一次QUICKSORT递归调用的期望运行时间，而不是执行的比较次数')
        print(' a) 给定一个大小为n的数组，任何特定元素被选为主元的概率为1/n')
        print('思考题7-3 Stooge排序')
        A = [8, 7, 56, 43, 21]
        print('数组A', _deepcopy(A), '的Stooge排序结果为:', stooge.stoogesort(A), A)
        print('思考题7-4 快速排序的堆栈深度')
        print(' 7.1中的快速排序算法包含有两个对其自身的递归调用,但是第二个递归不是必须的')
        A = [8, 7, 56, 43, 21]
        print('数组A', _deepcopy(A), '的尾递归快速排序结果为:', quicksort.morequicksort(A))
        print('思考题7-5 \"三数取中\"划分 也就是主元素RANDOMIZED-QUICKSORT的RANDOMIZED-PARTITION过程')
        print(' 三数取中方法仅仅影响其运行时间Ω(nlgn)中的常数因子')
        print('思考题7-6 对区间的模糊排序:算法的目标是对这些区间进行模糊排序')
        print('模糊排序算法的期望运行时间为Θ(nlgn),但当所有区间都重叠时，期望的运行时间为Θ(n)')
        # python src/chapter7/chapter7note.py
        # python3 src/chapter7/chapter7note.py

chapter7_1 = Chapter7_1()
chapter7_2 = Chapter7_2()
chapter7_3 = Chapter7_3()
chapter7_4 = Chapter7_4()

def printchapter7note():
    '''
    print chapter7 note.
    '''
    print('Run main : single chapter seven!')  
    chapter7_1.note()
    chapter7_2.note()
    chapter7_3.note()
    chapter7_4.note()

# python src/chapter7/chapter7note.py
# python3 src/chapter7/chapter7note.py
if __name__ == '__main__':  
    printchapter7note()
else:
    pass
