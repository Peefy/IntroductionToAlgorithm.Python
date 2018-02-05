
## python src/chapter_1/chapter1.py
## python3 src/chapter_1/chapter1.py

import sys
import numpy as nm
import matplotlib as mat
import matplotlib.pyplot as plt

class Chapter2:

    def __init__(self, ok = 1, *args, **kwargs):       
        '''
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        self

        Example
        =
        >>> chapter2 = Chapter2(ok = 1);
        '''
        self.ok = ok

    def note(self, *args, **kwargs):
        '''
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        self

        Example
        =
        >>> print('chapter1 note as follow:')
        '''  
        print('插入排序(INSERTION-SORT):输入n个数，输出n个数的升序或者降序排列')
        print('插入排序是一个对少量元素进行排序的有效算法，工作做原理与打牌摸牌整理手中的牌差不多')
        print('以下是Python的插入排序(升序)算法(模拟打牌)')
        print('书中的伪代码数组索引从1开始，python数组索引从0开始')
        A = [4, 4.5, 2, 5, 1.2, 3.5]
        print("待排序的序列：", A)
        n = len(A)
        for j in range(1, n):
            ## Insert A[j] into the sorted sequece A[1...j-1] 前n - 1 张牌
            # 下标j指示了待插入到手中的当前牌，所以j的索引从数组的第二个元素开始
            # 后来摸的牌
            key = A[j]
            # 之前手中的已经排序好的牌的最大索引
            i = j - 1
            # 开始寻找插入的位置并且移动牌
            while(i >= 0 and A[i] > key):
                # 移动牌
                A[i + 1] = A[i]
                # 遍历之前的牌
                i = i - 1
            # 后来摸的牌插入相应的位置
            A[i + 1] = key
        # 输出升序排序后的牌
        print("插入排序后的序列：", A)
        print('循环不变式主要用来帮助理解插入算法的正确性。证明循环不变式的三个性质')
        print(' 1.初始化：在循环的第一轮迭代开始前，应该是正确的')
        print(' 2.保持：如果在循环的某一次迭代开始之前它是正确的，那么在下一次迭代开始前，它也应该保持正确')
        print(' 3.终止：当循环结束时，不变式给了我们一个有用的性质，有助于表明算法是正确的')
        print('数学归纳法中，要证明某一性质是成立的，必须首先')
        return self

if __name__ == '__main__':
    print('single chapter two!')
    Chapter2().note()
    print('')

else:
    print('please in your cmd input as follow:\n python src/chapter_1/chapter1.py or \n' + 
        'python3 src/chapter_1/chapter1.py')
    print()

## python src/chapter_1/chapter1.py
## python3 src/chapter_1/chapter1.py

