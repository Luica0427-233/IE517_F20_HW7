# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:53:59 2020

@author: Lucia
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time


df = pd.read_csv("D:/Desktop/IE517 ML in Fin Lab/Module 7 - Ensembling for Performance Improvement/HW7/ccdefault.csv")

# Part 1: Random forest estimators

# Train_test split
X = df.iloc[:, 1:24].values
y = df.iloc[:,24]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


estimators = [1,5, 10, 50, 100, 200]
CV_train = []
CV_test = []

for i in estimators:

    start_time = time.process_time()
    
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(X_train_std, y_train)
    scores_CV_train = cross_val_score(forest, X_train_std, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    scores_CV_test = cross_val_score(forest, X_test_std, y_test, cv=10, scoring='accuracy', n_jobs=-1)

    end_time = time.process_time()

    # Calculate the mean
    mean_CV_train = scores_CV_train.mean()
    CV_train.append(mean_CV_train)
    mean_CV_test = scores_CV_test.mean()
    CV_test.append(mean_CV_test)

    print('N_estimators: ', i)
    print('In sample accuracy: ', mean_CV_train)
    print('Out of sample accuracy: ', mean_CV_test)
    print('Computation time: ', end_time-start_time, 's')
    print("-----------")

print("-----------")
print("CV_train:")
print(CV_train)
print("CV_test:")
print(CV_test)

# Part 2: Random forest feature importance

feature_labels = df.columns[1:-1]

forest = RandomForestClassifier(n_estimators=200)
forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
xlabel = []

for i in range(X_train_std.shape[1]):
    xlabel.append(feature_labels[indices[i]])
    print("%2d) %-*s %f" % (i+1, 30, feature_labels[indices[i]], importances[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(X_train_std.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train_std.shape[1]), xlabel, rotation=90)
plt.xlim([-1, X_train_std.shape[1]])
plt.tight_layout()
plt.show()



print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")