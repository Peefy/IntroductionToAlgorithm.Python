
# python src/chapter5/chapter5_4.py
# python3 src/chapter5/chapter5_4.py

import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy

class Chapter5_4:
    '''
    CLRS 第五章 5.4 算法函数和笔记
    '''

    def note(self):
        '''
        Summary
        =
        Print chapter5.4 note

        Example
        =
        >>> Chapter5_4().note()
        '''
        print('第五章 概率分析和随机算法')
        print('5.4 概率分析和指示器随机变量的进一步使用')
        print('5.4.1 生日悖论：一个房间里面的人数必须要达到多少。才能有两个人的生日相同的机会达到50%')
        print('出现的悖论就在于这个数目事实上远小于一年中的天数，甚至不足年内天数的一半')
        print('我们用整数1,2,...,k对房间里的人编号，其中k是房间里的总人数。另外不考虑闰年的情况，假设所有年份都有n=365天')
        print('而且假设人的生日均匀分布在一年的n天中，索引生日出现在任意一天的概率为1/n')
        print('两个人i和j的生日正好相同的概率依赖于生日的随机选择是否是独立的')
        print('i和j的生日都落在同一天r上的概率为1/n*1/n=1/n^2,所以两人同一天的概率为1/n')
        print('可以通过考察一个事件的补的方法，来分析k个人中至少有两人相同的概率')
        print('至少有两个人生日相同的概率=1-所有人生日都不相同的概率')
        print('所以问题转化为k个人所有人生日都不相同的概率小于1/2')
        print('k个人生日都不相同的概率为P=1*(n-1)/n*(n-2)/n*...*(n-k+1)/n')
        print('且由于1+x<=exp(x),P<=exp(-1/n)exp(-2/n)...exp(-(k-1)n)=exp(-k(k-1)/2n)<=1/2')
        print('所以当k(k-1)>=2nln2时，结论成立')
        print('所以当一年有n=365天时,至少有23个人在一个房间里面，那么至少有两个人生日相同的概率至少是1/2')
        print('当然如果是在火星上，一年有669个火星日，所以要达到相同效果必须有31个火星人')
        print('利用指示器随机变量，可以给出生日悖论的一个简单而近似的分析。对房间里k个人中的每一对(i,j),1<=i<j<=k')
        print('定义指示器随机变量Xij如果生日相同为1生日不同为0')
        print('根据引理5.1 E[Xij]=Pr{i和j生日相同}=1/n')
        print('令X表示计数至少具有相同生日的两人对数目的随机变量，得X=∑∑Xij,i=1 to n j = i + 1 to n')
        print('E[X]=∑∑E[Xij]=k(k-1)/2n,因此当k(k-1)>=2n时，有相同生日的两人对的对子期望数目至少是1个')
        print('如果房间里面至少有sqrt(2n)+1个人，就可以期望至少有两个人生日相同')
        print('对于n=365，如果k=28,具有相同生日的人的对子期望数值为(28*27)/(2*365)≈1.0356.因此如果至少有28个人')
        print('对于上述两种算法，第一种分析仅利用了概率，给出了为使存在至少一对人生日相同的概率大于1/2所需的人数')
        print('第二种分析使用了指示器随机变量，给出了所期望的相同生日数为1时的人数。虽然两种情况下的准确数目不等，但他们在渐进意义上是相等的，都是Θ(sqrt(n))')
        print('5.4.2 球与盒子')
        print('把相同的球随机投到b个盒子里的过程，其中盒子编号为1,2,...,b。每次投球都是独立的')
        # python src/chapter5/chapter5_4.py
        # python3 src/chapter5/chapter5_4.py
        return self

_instance = Chapter5_4()
note = _instance.note  

if __name__ == '__main__':  
    print('Run main : single chapter five!')  
    Chapter5_4().note()
else:
    pass
