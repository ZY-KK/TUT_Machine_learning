import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
figure, ax = plt.subplots(4, 1)
N = 100
n = np.arange(N)
x = np.arange(900)
y = np.cos(2*np.pi*0.1*n)
y = np.concatenate((np.zeros(500), y))
y = np.concatenate((y, np.zeros(300)))
# plot the noiseless signal
ax[0].plot(x, y, 'b-')

# plot the noisy signal
y_n = y+np.sqrt(0.5)*np.random.randn(y.size)
ax[1].plot(x, y_n, 'b-')

# Implement the deterministic sinusoid detector
T = stats.norm.ppf(0.17, loc=0, scale=0.5)
# f_0 = 0.015
f_0 = 1.0/T
cos_var = np.cos(2*np.pi*0.1*n)
y_deter_detector = np.convolve(cos_var, y_n, 'same')
ax[2].plot(x, y_deter_detector, 'b-')


#  random signal version

h = np.exp(-2 * np.pi * 1j * 0.1 * n)
y_random_dector = np.abs(np.convolve(h, y_n, 'same'))
ax[3].plot(x, y_random_dector, 'b-')
plt.show()
