#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:36:49 2018

@author: yihedeng
"""

from sklearn.ensemble import RandomForestClassifier

import csv
import sys
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

from sklearn.model_selection import cross_val_score

csv.field_size_limit(sys.maxsize)


clean = pd.read_csv('results.csv')



y = np.array(clean.iloc[:,-1]).reshape(-1, 1)
X = np.array(clean.drop(clean.columns[-1], 1))



clf = RandomForestClassifier(n_estimators=3000, max_depth=20,random_state=3)
clf.fit(X, y)

print(clf.feature_importances_)

yf = np.array(clean.iloc[:,-1])
y_pred = clf.predict(X)
pscore_train = metrics.accuracy_score(yf,y_pred)
print(pscore_train)

scores = cross_val_score(clf, X, y, cv=5)
print(scores)

