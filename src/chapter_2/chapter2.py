
## python src/chapter_1/chapter1.py
## python3 src/chapter_1/chapter1.py

import sys
import numpy as nm
import matplotlib as mat
import matplotlib.pyplot as plt

class Chapter2:

    def __init__(self, ok = 1, *args, **kwargs):
        self.ok = ok

    def note(*args, **kwargs):
        '''
        These are notes of Peefy CLRS chapter1

        Parameters
        =
        *args : a tuple like
        **kwargs : a dict like

        Returns
        =
        None

        Example
        =
        >>> print('chapter1 note as follow:')
        '''  
        print('插入排序(INSERTION-SORT):输入n个数，输出n个数的升序或者降序排列')
        print('')

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

