# python main.py
# python3 main.py

# Jupyter Notebook

import src

from src.test import testAll

from src.chapter1.chapter1_1 import Chapter1

from src.chapter2.chapter2 import Chapter2
from src.chapter2.chapter2_3 import Chapter2_3

from src.chapter3.chapter3_1 import Chapter3_1
from src.chapter3.chapter3_2 import Chapter3_2

from src.chapter4.chapter4_1 import Chapter4_1
from src.chapter4.chapter4_2 import Chapter4_2
from src.chapter4.chapter4_3 import Chapter4_3
from src.chapter4.chapter4_4 import Chapter4_4

from src.chapter5.chapter5_1 import Chapter5_1
from src.chapter5.chapter5_2 import Chapter5_2
from src.chapter5.chapter5_3 import Chapter5_3
from src.chapter5.chapter5_4 import Chapter5_4

from src.chapter6.chapter6_1 import Chapter6_1
from src.chapter6.chapter6_2 import Chapter6_2

if __name__ == '__main__':
    ## Test required packet
    testAll()
    ## Chapter 1
    Chapter1().note()
    ## Chapter 2 (2.1 2.2)
    Chapter2().note()
    ## Chapter 2 (2.3)
    Chapter2_3().note()
    ## Chapter 3 (3.1)
    Chapter3_1().note()
    ## Chapter 3 (3.2)
    Chapter3_2().note()
    ## Chapter 4 (4.1)
    Chapter4_1().note()
    ## Chapter 4 (4.2)
    Chapter4_2().note()
    ## Chapter 4 (4.3)
    Chapter4_3().note()
    ## Chapter 4 (4.4)
    Chapter4_4().note()
    ## Chapter 5 (5.1)
    Chapter5_1().note()
    ## Chapter 5 (5.2)
    Chapter5_2().note()
    ## Chapter 5 (5.3)
    Chapter5_3().note()
    ## Chapter 5 (5.4)
    Chapter5_4().note()
    ## Chapter 6 (6.1)
    Chapter6_1().note()
else:
    pass

# python main.py
# python3 main.py


