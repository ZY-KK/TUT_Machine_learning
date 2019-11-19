from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Dataset.GTSRB_subset.simplelbp import local_binary_pattern
from Dataset.GTSRB_subset.traffic_signs import load_data, extract_lbp_features
from sklearn.model_selection import cross_val_score
import numpy as np
X, Y = load_data("./Dataset/GTSRB_subset")

F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))
# estimator
KNN_model = KNeighborsClassifier()
LDA_model = LinearDiscriminantAnalysis()
SVC_model = SVC(kernel='linear', C=1)
LOGIC_model = LogisticRegression()
# KNN
KNN_model.fit(F, Y)
# LDA
LDA_model.fit(F, Y)
# SVC
SVC_model.fit(F, Y)
# LOGIC
LOGIC_model.fit(F, Y)

# cross_val_score
knn_score = cross_val_score(KNN_model, F, Y, cv=5)
lda_score = cross_val_score(LDA_model, F, Y, cv=5)
svc_score = cross_val_score(SVC_model, F, Y, cv=5)
logic_score = cross_val_score(LOGIC_model, F, Y, cv=5)

print("score:")
print(np.mean(knn_score))
print(np.mean(lda_score))
print(np.mean(svc_score))
print(np.mean(logic_score))
