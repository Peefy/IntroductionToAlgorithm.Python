# python main.py
# python3 main.py

# Jupyter Notebook

import src

from src.test import testAll

if __name__ == '__main__':
    ## Test required packet
    testAll()
    ## output all src files
    print(src.__all__)
    ## Chapter 1
    src.Chapter1().note()
    ## Chapter 2 (2.1 2.2)
    src.Chapter2().note()
    ## Chapter 2 (2.3)
    src.Chapter2_3().note()
    ## Chapter 3 (3.1)
    src.Chapter3_1().note()
    ## Chapter 3 (3.2)
    src.Chapter3_2().note()
    ## Chapter 4 (4.1)
    src.Chapter4_1().note()
    ## Chapter 4 (4.2)
    src.Chapter4_2().note()
    ## Chapter 4 (4.3)
    src.Chapter4_3().note()
    ## Chapter 4 (4.4)
    src.Chapter4_4().note()
    ## Chapter 5 (5.1)
    src.Chapter5_1().note()
    ## Chapter 5 (5.2)
    src.Chapter5_2().note()
    ## Chapter 5 (5.3)
    src.Chapter5_3().note()
    ## Chapter 5 (5.4)
    src.Chapter5_4().note()
    ## Chapter 6 
    src.chapter6_printall()
    ## chapter 7
    src.printchapter7note()
    ## chapter8
    src.printchapter8note()
    ## chapter9
    src.printchapter9note()
    ## chapter10
    src.printchapter10note()
    ## chapter11
    src.printchapter11note()
    ## chapter12
    src.printchapter12note()
else:
    pass

# python main.py
# python3 main.py


