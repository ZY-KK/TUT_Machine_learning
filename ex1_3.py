from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os


mat = loadmat("./Dataset/Ex1_data/twoClassData.mat")
print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()

X = np.array(X)
y = np.array(y)
x_0 = X[y == 0, :]
print(x_0)
# print(len(X))

# X[:, 0] = X[y == 0, :]


x_1 = X[y == 1, :]
# X[:, 1] = X[y == 1, :]

# X[:, 1] = x_1
plt.plot(x_0[:, 0], x_0[:, 1], 'ro')
plt.plot(x_1[:, 0], x_1[:, 1], 'bo')
# print(X[:, 0])
# plt.scatter(X[:, 0], X[:, 1], color=['red', 'blue'])
plt.show()
