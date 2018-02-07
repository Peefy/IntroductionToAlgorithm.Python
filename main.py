# python main.py
# python3 main.py
from src.numpyAndmatplot.test import testMatplot
from src.numpyAndmatplot.test import testNumpy
from src.chapter1.chapter1_1 import Chapter1
from src.chapter2.chapter2 import Chapter2
from src.chapter2.chapter2_3 import Chapter2_3

if __name__ == '__main__':
    ## Matplotlib and Numpy Test
    testNumpy()
    testMatplot()
    ## Chapter.1
    Chapter1().note()
    Chapter2().note()
    Chapter2_3().note()
else:
    pass

# python main.py
# python3 main.py


