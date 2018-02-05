
# python src/chapter_2/chapter2_3.py
# python3 src/chapter_2/chapter2_3.py 

class Chapter2_3:

    def note(self):
        '''
        Summary
        =
        Print chapter2.3 note
        Example
        =
        >>> Chapter2_3().note()
        '''
        print('chapter 2.3 note')
        print('算法设计有很多方法')
        print('如插入排序方法使用的是增量方法，在排好子数组A[1..j-1]后，将元素A[j]插入，形成排序好的子数组A[1..j]')
        print('2.3.1 分治法')
        print('很多算法在结构上是递归的，所以采用分治策略，将原问题划分成n个规模较小而结构与原问题相似的子问题，递归地解决这些子问题，然后再合并其结果，就得到原问题的解')
        print('分治模式在每一层递归上都有三个步骤：分解，解决，合并')
        print('合并排序')
        print('分解：将n个元素分成各含n/2个元素的子序列；')
        print('解决：用合并排序法对两个子序列递归地排序；')
        print('合并：合并两个已排序的子序列以得到排序结果')
        print('在对子序列排序时，其长度为1时递归结束。单个元素被视为是已经排序好的')
        print('合并排序的关键步骤在与合并步骤中的合并两个已排序子序列')

if __name__ == '__main__':
    Chapter2_3().note()
else:
    pass

# python src/chapter_2/chapter2_3.py
# python3 src/chapter_2/chapter2_3.py

