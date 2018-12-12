#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:14:37 2018

@author: yihedeng
"""

import csv
import sys
import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)


train = []
business = {}
users = {}

businessAvgStars = []
userAvgStars = []

testQueries = pd.read_csv('test_queries.csv')
        
with open('business.csv') as businessCsv:
    businessData = csv.DictReader(businessCsv)
    for row in businessData:
        i = row['business_id']
        business[i] = {}
        business[i]["averageStars"] = row['stars']
        business[i]["reviewCounts"] = row['review_count']
        
        #business.append(businessItem)
        #businessAvgStars.append(businessItem["averageStars"])
        
with open('users.csv') as usersCsv:
    userData = csv.DictReader(usersCsv)
    for row in userData:
        i = row['user_id']
        users[i] = {}
        users[i]["averageStars"] = row['average_stars']
        users[i]["reviewCounts"] = row['review_count']
        
        #users.append(userItem)
        #userAvgStars.append(userItem["averageStars"])
   
testFinal   = []   
test_business = testQueries["business_id"]
test_user = testQueries["user_id"]
for idx,val in enumerate(test_business):
    business_id = val
    user_id = test_user[idx]
    trainItem = {"userId":user_id,"businessId":business_id}
    
    trainItem["userAverageStars"] = users[user_id]['averageStars']
    trainItem["businessAverageStars"] = business[business_id]['averageStars']
    trainItem["userReviewCounts"] = users[user_id]['reviewCounts']
    trainItem["businessReviewCounts"] = business[business_id]['reviewCounts']
    
    testFinal.append(trainItem)

        
#print train
    
'''    
with open('train_reviews_clean.csv','w') as csvcleanfile:
    names = ["userId", "businessId", "stars"]
    writer = csv.DictWriter(csvcleanfile, fieldnames=names)
    writer.writeheader()
    for row in train:
        writer.writerow(row)
'''


with open('test_reviews_avgstars.csv','w') as csvfileAvg:
    names = ["userId", "businessId", "stars",
             "userAverageStars","businessAverageStars",
             "userReviewCounts", "businessReviewCounts"]
    writer = csv.DictWriter(csvfileAvg, fieldnames=names)
    writer.writeheader()
    for row in testFinal:
        writer.writerow(row)

        
    