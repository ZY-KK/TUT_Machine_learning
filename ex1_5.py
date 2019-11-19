import numpy as np
import matplotlib.pyplot as plt
import math

average = 0.0
for i in range(100):
    x = []
    f_0 = 0.015
    w = np.sqrt(0.3)*np.random.randn(100)
    x = math.sin(2*(math.pi)*f_0*i)+w

    plt.plot(x, 'bo')
    # plt.show()

    scores = []
    frequencies = []
    for f in np.linspace(0, 0.5, 1000):
        # Create vector e. Assume data is in x.
        n = np.arange(100)
        # <compute -2*pi*i*f*n. Imaginary unit is 1j>
        z = -2*math.pi*(1J)*f*n
        e = np.exp(z)
        # <compute abs of dot product of x and e>
        score = np.abs(np.dot(x, e))
        scores.append(score)
        frequencies.append(f)
    fHat = frequencies[np.argmax(scores)]
    average += fHat
    # print(fHat)
    """
    for i in range(100):
        w = np.sqrt(0.3)*np.random.randn(100)

        x.append(math.sin(2*(math.pi)*fHat*i)+w)
    """
    plt.plot(x, 'ro')
    # plt.show()

print(average/100)
