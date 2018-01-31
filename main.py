# python main.py
# python3 main.py

import sys
import numpy as nm
import numpy.fft as fft
import matplotlib.pyplot as plt

## totol test nm.array plot fft
X = nm.array([1, 2, 3, 4])
Q = nm.array([1, 2, 3, 4])
T = nm.arange(1, 100, 0.1)
S = nm.sin(T)
print(X)
plt.plot(T, S)
plt.figure()
freq = fft.fftfreq(T.shape[-1])
plt.plot(freq, fft.fft(S))
plt.show()

## Chapter.1


# python main.py
# python3 main.py


