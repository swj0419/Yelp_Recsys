#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:24:25 2018

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

test1 = pd.read_csv('simpleRegression.csv')
test2 = pd.read_csv('neuralNetwork.csv')
test3 = pd.read_csv('Boost.csv')

stars1 = test1['stars']
stars2 = test2['stars']
stars3 = test3['stars']

starsAll = (stars1 + stars2 + stars3) / 3.0


aff = []
for i in starsAll:
    if i < 1:
        i = 1
    if i > 5:
        i = 5
    aff.append(i) 
submission = pd.DataFrame({'stars': aff})
submission.index.name = "index"
submission.to_csv("average.csv")


