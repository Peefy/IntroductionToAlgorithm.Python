
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def fft_demo():
    x = np.arange(-100, 100, 0.5)
    y = np.sin(x) + np.sin(3 * x)
    plt.figure()
    plt.plot(x, y)
    plt.show()
    plt.figure()
    plt.plot(fft.fftfreq(x.shape[-1]), abs(fft.fft(y)))
    plt.show()
    plt.imshow(np.sin(np.outer(x, x)))
    plt.show()

if __name__ == '__main__':  
    fft_demo()
else:
    pass
