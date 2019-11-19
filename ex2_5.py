from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
# load the data
mat = loadmat("./Dataset/Ex1_data/twoClassData.mat")
X = mat["X"]
Y = mat["y"].ravel()

# split the sample
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

# KNN
model = KNeighborsClassifier()
model.fit(x_train, y_train)

# predict
knn_predict_y = model.predict(x_test)
# print(test_y)

knn_score = accuracy_score(y_test, knn_predict_y.ravel(), normalize=True, sample_weight=None)
print("knn-score: %f" % (knn_score))

# LDA
clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
lda_predict_y = clf.predict(x_test)

lda_score = accuracy_score(y_test, lda_predict_y)
print("lda-score: %f" % (lda_score))
