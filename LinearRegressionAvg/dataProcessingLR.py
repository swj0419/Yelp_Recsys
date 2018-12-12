#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:34:46 2018

@author: yihedeng
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:17:59 2018

@author: yihedeng
"""
import csv
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)


train = []
business = []
users = []

businessAvgStars = []
userAvgStars = []

with open('train_reviews.csv') as csvfile:
    trainData = csv.DictReader(csvfile)
    for row in trainData:
        #date = row[]
        trainItem = {"userId": "",
         "businessId": "", 
         "stars": 0.0}
        
        business_id = row['business_id']
        user_id = row['user_id']
        star = row['stars']
        #trainItem["isCool"] = row['cool']
        #trainItem["isFunny"] = row['funny']
        #trainItem["isUseful"] = row['useful']
        
        trainItem["businessId"] = business_id
        trainItem["userId"] = user_id
        trainItem["stars"]=float(star)
        
        train.append(trainItem)
        
with open('business.csv') as businessCsv:
    businessData = csv.DictReader(businessCsv)
    for row in businessData:
        businessItem = {"businessId": "", 
                        "averageStars": 0.0}
        businessItem["businessId"] = row['business_id']
        businessItem["averageStars"] = row['stars']
        businessItem["reviewCounts"] = row['review_count']
        
        business.append(businessItem)
        #businessAvgStars.append(businessItem["averageStars"])
        
with open('users.csv') as usersCsv:
    userData = csv.DictReader(usersCsv)
    for row in userData:
        userItem = {"userId": "", 
                    "averageStars": 0.0}
        userItem["userId"] = row['user_id']
        userItem["averageStars"] = row['average_stars']
        userItem["reviewCounts"] = row['review_count']
        
        users.append(userItem)
        #userAvgStars.append(userItem["averageStars"])
   
trainFinal   = []   
for trainItem in train:
    business_id = trainItem["businessId"]
    user_id = trainItem["userId"]
    
    thisUser = (itemU for itemU in users if itemU["userId"] == user_id).next()
    thisBusiness = (itemB for itemB in business if itemB["businessId"] == business_id).next()
    trainItem["userAverageStars"] = thisUser['averageStars']
    trainItem["businessAverageStars"] = thisBusiness['averageStars']
    trainItem["userReviewCounts"] = thisUser['reviewCounts']
    trainItem["businessReviewCounts"] = thisBusiness['reviewCounts']
    
    trainFinal.append(trainItem)

        
#print train
    
'''    
with open('train_reviews_clean.csv','w') as csvcleanfile:
    names = ["userId", "businessId", "stars"]
    writer = csv.DictWriter(csvcleanfile, fieldnames=names)
    writer.writeheader()
    for row in train:
        writer.writerow(row)
'''


with open('train_reviews_avgstars.csv','w') as csvfileAvg:
    names = ["userId", "businessId", "stars",
             "userAverageStars","businessAverageStars",
             "userReviewCounts", "businessReviewCounts"]
    writer = csv.DictWriter(csvfileAvg, fieldnames=names)
    writer.writeheader()
    for row in trainFinal:
        writer.writerow(row)

        
    