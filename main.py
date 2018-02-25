# python main.py
# python3 main.py
from src.test import testAll
from src.chapter1.chapter1_1 import Chapter1
from src.chapter2.chapter2 import Chapter2
from src.chapter2.chapter2_3 import Chapter2_3
from src.chapter3.chapter3_1 import Chapter3_1

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
else:
    pass

# python main.py
# python3 main.py


