from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
digits = load_digits()
# print the digits keys
print(digits.target)
# plot the figure
plt.gray()
plt.imshow(digits.images[0])
plt.show()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
KNN_model = KNeighborsClassifier()
LDA_model = LinearDiscriminantAnalysis()
SVC_model = SVC(kernel='linear', C=1)
LOGIC_model = LogisticRegression()

for i in range(10):

    # KNN
    KNN_model.fit(x_train, y_train)
    # LDA
    LDA_model.fit(x_train, y_train)
    # SVC
    SVC_model.fit(x_train, y_train)
    # LOGIC
    LOGIC_model.fit(x_train, y_train)

# KNN predict
knn_predict = KNN_model.predict(x_test)
# LDA predict
lda_predict = LDA_model.predict(x_test)
# SVC predict
svc_predict = SVC_model.predict(x_test)
# LOGIC predict
logic_predict = LOGIC_model.predict(x_test)

# accuracy
# # KNN score
knn_score = accuracy_score(y_test, knn_predict)
print("KNN score: %f" % (knn_score))
lda_score = accuracy_score(y_test, lda_predict)
print("LDA score: %f" % (lda_score))
svc_score = accuracy_score(y_test, svc_predict)
print("SVC score: %f" % (svc_score))
logic_score = accuracy_score(y_test, logic_predict)
print("Logic score: %f" % (logic_score))


