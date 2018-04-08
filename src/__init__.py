'''
This packet includes CLRS notes of each chapter
'''

__all__ = ['Chapter1', 'Chapter2', 'Chapter2_3',
           'Chapter3_1', 'Chapter3_2', 'Chapter4_1', 
           'Chapter4_2', 'Chapter4_3', 'Chapter4_4', 
           'Chapter5_1', 'Chapter5_2', 'Chapter5_3', 
           'Chapter5_4', 'chapter6_printall',
           'printchapter7note',
           'printchapter8note',
           'printchapter9note',
           'printchapter10note',
           'printchapter11note',
           'printchapter12note',
           'printchapter13note']

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