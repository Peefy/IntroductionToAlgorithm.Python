
import sys
import numpy as nm
from numpy import arange
import numpy.fft as fft
import matplotlib.pyplot as plt

def testNumpy():
    '''
    Test numpy package fft show plot

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
    =
    None

    Return:
    =
    None

    Example:
    >>> test.testMatplot()
    '''
    ## totol test nm.array plot fft
    X = nm.array([1, 2, 3, 4])
    Q = nm.array([1, 2, 3, 4])
    T = nm.arange(1, 100, 0.1)
    S = nm.sin(T)
    plt.plot(T, S)
    plt.show()
    plt.figure()
    freq = fft.fftfreq(T.shape[-1])
    plt.plot(freq, fft.fft(S))
    plt.show()
