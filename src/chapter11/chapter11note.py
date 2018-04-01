
# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
'''
Class Chapter11_1

'''
import sys as _sys
import math as _math
import random as _random
import time as _time
from random import randint as _randint
from copy import copy as _copy, deepcopy as _deepcopy
from numpy import arange as _arange
import numpy as np

class Chapter11_1:
    '''
    chpater11.1 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.1 note

        Example
        ====
        ```python
        Chapter11_1().note()
        ```
        '''
        print('chapter11.1 note as follow')
        print('第11章 散列表')
        print('在很多应用中，都要用到一种动态集合结构，它仅支持INSERT,SEARCH的DELETE字典操作')
        print('实现字典的一种有效数据结构为散列表(HashTable)')
        print('在最坏情况下，在散列表中，查找一个元素的时间在与链表中查找一个元素的时间相同')
        print('在最坏情况下都是Θ(n)')
        print('但在实践中，散列技术的效率是很高的。在一些合理的假设下，在散列表中查找一个元素的期望时间为O(1)')
        print('散列表是普通数组概念的推广，因为可以对数组进行直接寻址，故可以在O(1)时间内访问数组的任意元素')
        print('如果存储空间允许，可以提供一个数组，为每个可能的关键字保留一个位置，就可以应用直接寻址技术')
        print('当实际存储的关键字数比可能的关键字总数较小时，这是采用散列表就会较直接数组寻址更为有效')
        print('在散列表中，不是直接把关键字用作数组下标，而是根据关键字计算出下标。')
        print('11.2着重介绍解决碰撞的链接技术。')
        print('所谓碰撞，就是指多个关键字映射到同一个数组下标位置')
        print('11.3介绍如何利用散列函数，根据关键字计算出数组的下标。')
        print('11.4介绍开放寻址法，它是处理碰撞的另一种方法。散列是一种极其有效和实用的技术，基本的字典操作只需要O(1)的平均时间')
        print('11.5解释当待排序的关键字集合是静态的，\"完全散列\"如何能够在O(1)最坏情况时间内支持关键字查找')
        print('11.1 直接寻址表')
        print('当关键字的全域U比较小时，直接寻址是一种简单而有效的技术。假设某应用要用到一个动态集合')
        print('其中每个元素都有一个取自全域U的关键字，此处m是一个不很大的数。另外假设没有两个元素具有相同的关键字')
        print('为表示动态集合，用一个数组(或直接寻址表)T[0...m-1],其中每个位置(或称槽)对应全域U中的一个关键字')
        print('对于某些应用，动态集合中的元素可以放在直接寻址表中。亦即不把每个元素的关键字及其卫星数据都放在直接寻址表外部的一个对象中')
        print('但是，如果不存储关键字，就必须有某种办法来确定某个槽是否为空')
        print('练习11.1-1: 考虑一个由长度为m的直接寻址表T表示的动态集合S。给出一个查找S的最大元素的算法过程')
        print(' 所给的过程在最坏情况下的运行时间是O(m)')
        print('练习11.1-2: 位向量(bit vector)是一种仅包含0和1的数组。长度为m的位向量所占空间要比包含m个指针的数组少得多')
        print(' 请说明如何用一个位向量来表示一个包含不同元素的动态集合。字典操作的运行时间应该是O(1)')
        print('练习11.1-3: 说明如何实现一个直接寻址表，使各元素的关键字不必都相同，且各元素可以有卫星数据。')
        print('练习11.1-4: 希望通过利用一个非常大的数组上直接寻址的方式来实现字典')
        print(' 开始时，该数组中可能包含废料，但要对整个数组进行初始化是不实际的，因为该组的规模太大')
        print(' 请给出在大数组上实现直接寻址字典的方案。每个存储的对象占用O(1)空间')
        print(' 操作SEARCH,INSERT和DELETE的时间为O(1),对数据结构初始化的时间O(1)')
        print(' 可以利用另外一个栈，其大小等于实际存储在字典中的关键字数目，以帮助确定大型数组中某个给定的项是否是有效的')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_2:
    '''
    chpater11.2 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.2 note

        Example
        ====
        ```python
        Chapter11_2().note()
        ```
        '''
        print('chapter11.2 note as follow')
        print('11.2 散列表')
        print('直接寻址技术存在着一个明显的问题：如果域U很大，',
            '在一台典型计算机的可用内存容量限制下，要在机器中存储大小为U的一张表T就有点不实际甚至是不可能的了')
        print('实际要存储的关键字集合K相对于U来说可能很小，因而分配给T的大部分空间都要浪费掉')
        print('当存储在字典中的关键字集合K比所有可能的关键字域U要小的多时，散列表需要的存储空间要比直接寻址少很多')
        print('特别地，在保持仅需O(1)时间即可在散列表中查找一个元素的好处情况下，存储要求可以降至Θ(|K|)')
        print('在直接寻址方式下，具有关键字k的元素被存放在槽k中。在散列方式下，该元素处于h(k)中')
        print('亦即，利用散列函数h,根据关键字k计算出槽的位置。函数h将关键字域U映射懂啊散列表T[0..m-1]的槽位上：')
        print('这时，可以说一个具有关键字k到元素是被散列在槽h(k)上，或说h(k)是关键字k的散列值')
        print('两个关键字可能映射到同一个槽上。将这种情形称为发生了碰撞')
        print('当然，最理想的解决方法是完全避免碰撞')
        print('可以考虑选用合适的散列函数h。在选择时有一个主导思想，就是使h尽可能地\"随机\",从而避免或至少最小化碰撞')
        print('当然，一个散列函数h必须是确定的，即某一给定的输入k应始终产生相同的结果h(k),')
        print('通过链接法解决碰撞')
        print(' 在链接法中，把散列到同一槽中的所有元素都放在同一个链表中，槽j中有一个指针，它指向由所有散列到j的元素构成的链表的头')
        print(' 插入操作的最坏情况运行时间为O(1).插入过程要快一些，因为假设要插入的元素x没有出现在表中；如果需要，在插入前执行搜索，可以检查这个假设(付出额外代价)')
        print('CHAINED-HASH-INSERT(T, x)')
        print(' insert x at the head of list T[h(key[x])]')
        print('CHAINED-HASH-SEARCH(T, x)')
        print(' search for an element with k in list T[h(k)]')
        print('CHAINED-HASH-DELETE(T, x)')
        print(' delete x from the list T[h(key[x])]')
        print('对用链接法散列的分析')
        print(' 采用链接法后散列的性能怎样呢？特别地，要查找一个具有给定关键字的原宿需要多长时间呢？')
        print(' 给定一个能存放n个元素的，具有m个槽位的散列表T，定义T的装载因子a为n/m,即一个链表中平均存储的元素数')
        print(' 分析以a来表达，a可以小于、等于或者大于1')
        print('用链接法散列的最坏情况性能很差；所有的n个关键字都散列到同一个槽中，从而产生出一个长度为n的链表')
        print('最坏情况下查找的时间为Θ(n),再加上计算散列函数的时间，这么一来就和用一个链表来来链接所有的元素差不多了。显然，')
        print('散列方法的平均性态依赖于所选取的散列函数h在一般情况下，将所有的关键字分布在m个槽位上的均匀程度')
        print('先假定任何元素散列到m个槽中每一个的可能性是相同的，且与其他元素已被散列到什么位置上是独立无关的')
        print('称这个假设为简单一致散列')
        print('假定可以在O(1)时间内计算出散列值h(k),从而查找具有关键字为k的元素的时间线性地依赖于表T[h(k)]的长度为n')
        print('先不考虑计算散列函数和寻址槽h(k)的O(1)时间')
        print('定理11.1 对一个用链接技术来解决碰撞的散列表，在简单一致散列的假设下，一次不成功查找期望时间为Θ(1+a)')
        print('定理11.2 在简单一致散列的假设下，对于用链接技术解决碰撞的散列表，平均情况下一次成功的查找需要Θ(1+a)时间')
        print('练习11.2-1: 假设用一个散列函数h，将n个不同的关键字散列到一个长度为m的数组T中。')
        print(' 假定采用的是简单一致散列法，那么期望的碰撞数是多少？')
        print('练习11.2-2: 对于一个利用链接法解决碰撞的散列表，说明将关键字5,28,19,15,20,33,12,17,10')
        print(' 设该表中有9个槽位，并设散列函数为h(k)=k mod 9')
        print('练习11.2-3: 如果将链接模式改动一下，使得每个链表都能保持已排序顺序，散列的性能就可以有很大的提高。')
        print(' 这样的改动对成功查找、不成功查找、插入和删除操作的运行时间有什么影响')
        print('练习11.2-4: 在散列表内部，如何通过将所有未占用的槽位链接成一个自由链表，来分配和去分配元素的存储空间')
        print(' 假定一个槽位可以存储一个标志、一个元素加上一个或两个指针')
        print(' 所有的字典和自由链表操作应具有O(1)的期望运行时间')
        print('练习11.2-5: 有一个U的大小为n的子集，它包含了均散列到同一个槽位中的关键字，这样对于带链接的散列表，最坏情况下查找时间为Θ(n)')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_3:
    '''
    chpater11.3 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.3 note

        Example
        ====
        ```python
        Chapter11_3().note()
        ```
        '''
        print('chapter11.3 note as follow')
        print('11.3 散列函数')
        print('好的散列函数的特点')
        print('一个好的散列函数应近似地满足简单一致散列的假设：每个关键字都等可能地散列到m个槽位的任何一个之中去')
        print('并与其它的关键字已被散列到哪一个槽位中无关')
        print('不幸的是：一般情况下不太可能检查这一条件是否成立，因为人们很少可能知道关键字所符合的概率分布，而各关键字可能并不是完全互相独立的')
        print('有时也能知道关键字的概率分布。例如：已知各关键字都是随机的实数k，独立地、一致地分布于范围[0,1)')
        print('在实践中，常常可以运用启发式技术来构造好的散列函数')
        print('例如，在一个编译器的符号表中，关键字都是字符串，表示程序中的标识符')
        print('同一个程序中，经常会出现一些很相近的符号，如pt和pts。')
        print('一个好的散列函数应能最小化将这些相近符号散列到同一个槽中的可能性')
        print('\"除法散列\"用一个特定的质数来除所给的关键字，所得的余数即为该关键字的散列值')
        print('假定所选择的质数与关键字分布中的任何模式都是无关的，这种方法常常可以给出很好的结果')
        print('散列函数的某些应用可能会要求比简单一致散列更强的性质，例如可能希望某些很近似的关键字具有截然不同的散列值')
        print('将关键字解释为自然数')
        print(' 如果所给关键字不是自然数，则必须有一种方法来将它们解释为自然数')
        print(' 标识符pt可以被解释为十进制整数对(112,116),pt即为(112*128)+116=14452')
        print('11.3.1 除法散列法')
        print(' 通过取k除以m的余数，来将关键字k映射到m个槽的摸一个中去，亦即散列函数为h(k) = k mod m')
        print(' 例如，如果散列表的大小为m=12,所给关键字为k=100,则h(k)=4。这种方法只要一次除法操作，所以比较快')
        print(' 应用除法散列时，要注意m的选择，m不应是2的幂；可以选作m的值常常是与2的整数幂不太接近的质数')
        print(' 例如，假设我们要分配一张散列表，并用链接法解决碰撞，表中大约要存放n=2000个字符串，每个字符有8位')
        print(' 一次不成功的查找大约要检查3个元素，但我们并不在意，故分配散列表的大小为m=701.')
        print(' 之所以选择701这个数，是因为它是个接近a=2000/3，但又不接近2的任何幂次的质数。把每个关键字k视为一个整数')
        print(' 则有散列函数h(k) = k mod 701')
        print('11.3.2 乘法散列法')
        print(' 构造散列函数的乘法方法包含两个步骤。第一步，用关键字k乘上常数A(0<A<1),', 
            '并抽出kA的小数部分。然后，用m乘以这个值，再取结果的底(floor)。总之，散列函数为h(k)=[m(kA mod 1)]')
        print(' 其中kA mod 1 即kA的小数部分，亦即kA-[kA]')
        print(' 乘法方法的一个优点是对m的选择没有特别的要求，一般选择它为2的某个幂次m=2^p')
        print(' 虽然这个方法对任何的A值都适用，但对某些值效果更好。Knuth认为最佳的选择与待散列的数据的特征有关A=(sqrt(5)-1)/2=0.6180339887...就是一个比较理想的值')
        print(' 例子：假设有k=123456,p=14,m=2^14=16384,w=32,根据Knuth的建议')
        print(' 取A为形如s/2^32的分数，它与(sqrt(5)-1)/2最为接近，于是A=2654435769/2^32')
        print(' k*s=32770622297664=(76300*2^32)+17612864,从而有r1=76300,r0=17612864,r0的14个最高有效位产生了散列值h(k)=67')
        print('11.3.3 全域散列')
        print('如果让某个与你作对的人来选择要散列的关键字，那么他会选择全部散列到同一槽中的n个关键字，使得平均检索值为Θ(n)')
        print('任何一个特定的散列函数都可能出现这种最坏情况性态：唯一有效的改进方法是随机地选择散列函数，使之独立要存储的关键字。')
        print('这种方法称作全域散列(universal hashing),不管对手选择了怎样的关键字，其平均性态都很好')
        print('全域散列的基本思想是在执行开始时，就从一族仔细设计的函数中，随机地选择一个座位散列函数')
        print('就像在快速排序中一样，随机化保证了没有哪一种输入会导致最坏情况性态。')
        print('同时，随机化使得即使对同一个输入，算法在每一次执行时的性态也都不一样')
        print('这样就可以确保对于任何输入，算法都具有较好的平均情况性态')
        print('设H为有限的一组散列函数，它将给定的关键字域U映射到{0,1,..,m-1}中。这样的一个函数组称为是全域的')
        print('定理11.3 如果h选自一组全域的散列函数，并用于将n个关键字散列到一个大小为m的、用链接法解决碰撞的表T中')
        print('对于每一个关键字k，定义一个随机变量Yk,它等于非k的、与k散列到同一槽位中的其他关键字的数目')
        print('推论11.4 对于一个具有m个槽位的表，利用全域散列和链接法解决碰撞，需要Θ(n)的期望时间来处理任何包含了n')
        print(' 个INSERT,SEARCH和DELETE操作的操作序列，该序列中包含了O(m)个INSERT操作')
        print('证明：由于插入操作的数目为O(m),有n=O(m),从而a=O(1),')
        print('')
        print('')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_4:
    '''
    chpater11.4 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.4 note

        Example
        ====
        ```python
        Chapter11_4().note()
        ```
        '''
        print('chapter11.4 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

class Chapter11_5:
    '''
    chpater11.5 note and function
    '''
    def note(self):
        '''
        Summary
        ====
        Print chapter11.5 note

        Example
        ====
        ```python
        Chapter11_5().note()
        ```
        '''
        print('chapter11.5 note as follow')
        # python src/chapter11/chapter11note.py
        # python3 src/chapter11/chapter11note.py

chapter11_1 = Chapter11_1()
chapter11_2 = Chapter11_2()
chapter11_3 = Chapter11_3()
chapter11_4 = Chapter11_4()
chapter11_5 = Chapter11_5()

def printchapter11note():
    '''
    print chapter11 note.
    '''
    print('Run main : single chapter eleven!')  
    chapter11_1.note()
    chapter11_2.note()
    chapter11_3.note()
    chapter11_4.note()
    chapter11_5.note()

# python src/chapter10/chapter10note.py
# python3 src/chapter10/chapter10note.py
if __name__ == '__main__':  
    printchapter11note()
else:
    pass
