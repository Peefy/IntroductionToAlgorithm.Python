'''
This packet includes CLRS notes of each chapter
'''

LAST_CHAPTER = 19

__all__ = ['Chapter1', 'Chapter2', 'Chapter2_3',
           'Chapter3_1', 'Chapter3_2', 'Chapter4_1', 
           'Chapter4_2', 'Chapter4_3', 'Chapter4_4', 
           'Chapter5_1', 'Chapter5_2', 'Chapter5_3', 
           'Chapter5_4', 'chapter6_printall']

for i in range(7, LAST_CHAPTER + 1):
    __all__.append('printchapter{}note'.format(i))

from .chapter1.chapter1_1 import Chapter1

from .chapter2.chapter2 import Chapter2
from .chapter2.chapter2_3 import Chapter2_3

from .chapter3.chapter3_1 import Chapter3_1
from .chapter3.chapter3_2 import Chapter3_2

from .chapter4.chapter4_1 import Chapter4_1
from .chapter4.chapter4_2 import Chapter4_2
from .chapter4.chapter4_3 import Chapter4_3
from .chapter4.chapter4_4 import Chapter4_4

from .chapter5.chapter5_1 import Chapter5_1
from .chapter5.chapter5_2 import Chapter5_2
from .chapter5.chapter5_3 import Chapter5_3
from .chapter5.chapter5_4 import Chapter5_4

from .chapter6.test import chapter6_printall

from .chapter7.chapter7note import printchapter7note

from .chapter8.chapter8note import printchapter8note

from .chapter9.chapter9note import printchapter9note

from .chapter10.chapter10note import printchapter10note

from .chapter11.chapter11note import printchapter11note

from .chapter12.chapter12note import printchapter12note

from .chapter13.chapter13note import printchapter13note

from .chapter14.chapter14note import printchapter14note

from .chapter15.chapter15note import printchapter15note

from .chapter16.chapter16note import printchapter16note

from .chapter17.chapter17note import printchapter17note

from .chapter18.chapter18note import printchapter18note

from .chapter19.chapter19note import printchapter19note
