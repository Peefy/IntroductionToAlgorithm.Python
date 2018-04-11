
import math
import sys
import numpy as np
from numpy import arange
import numpy.fft as fft
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from dugulib import sort
else:
    from .dugulib import sort

def testAll():
    '''
    Test required packet
    '''
    testNumpy()
    testMatplot()
    testSympy()

def testNumpy():
    '''
    Test numpy package arange

    Args:
    =
    None

    Return:
    =
    None

    Example:
    >>> test.testNumpy()
    '''
    print('numpy.arange:', arange(1, 10, 0.8))

def testMatplot():
    '''
    Test matplotlib package fft show plot

    Args:
    ===
    None

    Return:
    ===
    None

    Example:
    ===
    ```python
    >>> test.testMatplot()
    ```
    '''
    ## totol test nm.array plot fft
    X = np.array([1, 2, 3, 4])
    Q = np.array([1, 2, 3, 4])
    T = np.arange(1, 100, 0.1)
    S = np.sin(T)
    plt.plot(T, S)
    plt.show()
    plt.figure()
    freq = fft.fftfreq(T.shape[-1])
    plt.plot(freq, fft.fft(S))
    plt.show()

def testSympy():
    '''
    test sympy
    '''
    pass

if __name__ == '__main__':
    print(math.log2(3))
    print(math.log2(7))
    sort.test()
    
# python src/test.py
# python3 src/test.py