#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:26:17 2018

@author: yihedeng
"""

import csv
import sys
import numpy as np
import pandas as pd
import math
csv.field_size_limit(sys.maxsize)


train = []
business = []
users = []

businessAvgStars = []
userAvgStars = []

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


train = pd.read_csv('HTrainLast.csv')



new = pd.DataFrame()
for key,item in train.items():
    L  = []
    for entry in item:
        if pd.isnull(entry):
            entry = "Nothing"
        L.append(str(entry))
    new[key] = L
            

for key in new.keys():
    new[key] = le.fit_transform(new[key])

        

new.to_csv('results.csv', index=False, header=False)