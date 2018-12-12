#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:14:12 2018

@author: yihedeng
"""

import csv
import sys
import matplotlib.pylab as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

csv.field_size_limit(sys.maxsize)
train = []

'''
with open('train_reviews_avgstars.csv') as csvfile:
    trainData = csv.DictReader(csvfile)
    for row in trainData:
        #date = row[]
        trainItem = {"userId": "",
         "businessId": "", 
         "stars": 0.0,
         "userAverageStars": 0.0,
         "businessAverageStars": 0.0}
        
        business_id = row['businessId']
        user_id = row['userId']
        star = row['stars']
        userAverageStars = row['userAverageStars']
        businessAverageStars = row['businessAverageStars']
        
        trainItem["businessId"] = business_id
        trainItem["userId"] = user_id
        trainItem["stars"] = float(star)
        trainItem["userAverageStars"] = float(userAverageStars)
        trainItem["businessAverageStars"] = float(businessAverageStars)
        
        train.append(trainItem)
  '''


trainData = pd.read_csv('train_reviews_avgstars.csv')
testData = pd.read_csv('test_reviews_avgstars.csv')

averageAll = float(sum(trainData['stars'])) / float(len(trainData['stars']))
averageAllFeature = np.array([averageAll]*len(trainData['stars']))
differenceUserFeature = trainData['userAverageStars'] - averageAll
differenceBusinessFeature = trainData['businessAverageStars'] - averageAll

averageAllCounts = sum(list(trainData['userReviewCounts'])) / float(len(list(trainData['userReviewCounts'])))
averageAllCountsbus = sum(list(trainData['businessReviewCounts'])) / float(len(list(trainData['businessReviewCounts'])))


X_train = np.array(list(zip(
            differenceUserFeature*trainData['userReviewCounts'], 
            differenceBusinessFeature*trainData['businessReviewCounts'],
            trainData['userAverageStars'] * trainData['businessAverageStars'])))

X_test = np.array(list(zip(
            (testData['userAverageStars']- averageAll)*testData['userReviewCounts'], 
            (testData['businessAverageStars']- averageAll)*testData['businessReviewCounts'],
            testData['userAverageStars'] * testData['businessAverageStars'])))

y_train = np.array(trainData['stars'])
reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_train)
plt.plot(y_train[0:1000], y_pred[0:1000], '.')

print("reg: " + str(reg.score(X_train, y_train)))
print("reg: " + str(reg.coef_))
#y_test = reg.predict(X_test)
#y_test_modified = []


model = MLPRegressor(hidden_layer_sizes=(5,7),
                                     activation='relu',
                                     solver='adam',
                                     learning_rate='adaptive',
                                     max_iter=10000,
                                     learning_rate_init=0.001,
                                     alpha=0.001)
model.fit(X_train, y_train)
y_pred_N = model.predict(X_train)
plt.plot(y_train[0:1000], y_pred_N[0:1000], '.')
print("reg: " + str(reg.score(X_train, y_train)))
print("model: " + str(model.score(X_train, y_train)))


y_test = reg.predict(X_test)
y_test_modified = []
for y in y_test:
    if y > 5:
        y = 5.0
    if y < 1:
        y = 1.0
    y_test_modified.append(y)
    
y_test_nn = model.predict(X_test)
y_test_modified_nn = []
for y in y_test_nn:
    if y > 5:
        y = 5.0
    if y < 1.0:
        y = 1.0
    y_test_modified_nn.append(y)


df = pd.DataFrame({'stars':y_test_modified})
df.index.name = "index"
df.to_csv('simpleRegression.csv')

dfnn = pd.DataFrame({'stars':y_test_modified_nn})
dfnn.index.name = "index"
dfnn.to_csv('neuralNetwork.csv')
   
'''X_train, X_test, y_train, y_test = train_test_split([trainData['userAverageStars']], 
                                                    trainData['stars'], 
                                                    test_size=0.2, 
                                                    random_state=0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
'''







