from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from Dataset.GTSRB_subset.simplelbp import local_binary_pattern
from Dataset.GTSRB_subset.traffic_signs import load_data, extract_lbp_features
from sklearn.model_selection import cross_val_score
import numpy as np
X, Y = load_data("./Dataset/GTSRB_subset")
F = extract_lbp_features(X)
# 100-tree Random Forest classifier
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(F, Y)
# # rfc score
rfc_score = cross_val_score(rfc_model, F, Y, cv=5)


# 100-tree Extremely Randomized Trees classifier
etc_model = ExtraTreesClassifier(n_estimators=100)
etc_model.fit(F, Y)
# # etc score
etc_score = cross_val_score(etc_model, F, Y, cv=5)


# 100-tree AdaBoost classifier
abc_model = AdaBoostClassifier(n_estimators=100)
abc_model.fit(F, Y)
# # abc score
abc_score = cross_val_score(abc_model, F, Y, cv=5)


# 100-tree Gradient Boosted Tree classifier
gbc_model = GradientBoostingClassifier(n_estimators=100)
gbc_model.fit(F, Y)
# # gbc score
gbc_score = cross_val_score(gbc_model, F, Y, cv=5)


# # print score
print("score: ")
print(np.mean(rfc_score))
print(np.mean(etc_score))
print(np.mean(abc_score))
print(np.mean(gbc_score))
