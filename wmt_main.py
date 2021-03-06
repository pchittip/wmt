"""
===============================================================================
Walmart - Store Sales Forecasting
http://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
Use historical markdown data to predict store sales 
===============================================================================
"""
print(__doc__)

import pandas as pd
import matplotlib.pyplot as plt
import wmt_learn
import numpy as np


print '-'*80
print "\nTraining Set "
print wmt_learn.train.head()
print '-'*80

print "\nTest Set "
print wmt_learn.test.head()
print '-'*80

print "\nStore Data "
print wmt_learn.stores.head()
print '-'*80

print "\nFeature Data "
print wmt_learn.features.head()
print '-'*80

#Data Exploration...

#Feature Matrix 8190 rows x 12 cols
X_FeatureLabels = wmt_learn.features.columns
#X = wmt_learn.features.values

STORES = wmt_learn.stores
TRAIN = wmt_learn.train
TEST = wmt_learn.test

# Dictionary for changeing store type into scalar
stype = {'A': 0, 'B': 1, 'C': 2}


# Setting up data for input into learning algorithm
A=np.zeros(45)
X=np.identity(45)
X_Store = np.vstack((A, X[X[:,0] < 45]))

X_Type = np.identity(3)

A=np.zeros(99)
X=np.identity(99)
X_Dept = np.vstack((A, X[X[:,0] < 99]))

X_size = np.zeros(1)

A=np.zeros(53)
X=np.identity(53)
X_Week = np.vstack((A, X[X[:,0] < 53]))



#for row in TRAIN.index:
#    print TRAIN.ix[row]
X_input = []   
Y_output = [] 

print 'Pre-Processing Data...'
for row in np.arange(421570):
    store_num = TRAIN.ix[row].Store
#    print X_Store[store_num]
    store_tpe = stype[STORES[STORES.Store==store_num].Type.values[0]]
#    print X_Type[store_tpe]
    store_dept = TRAIN.ix[row].Dept
#    print X_Dept[store_dept]
#    print X_Week[TRAIN.ix[row].Date.week]
#    print TRAIN.ix[row].IsHoliday*1.
    X = np.concatenate([X_Store[store_num],
                        X_Type[store_tpe],
                        X_Dept[store_dept],
                        X_Week[TRAIN.ix[row].Date.week],
                        [TRAIN.ix[row].IsHoliday*1.]])
    X_input += [X]
    Y_output += [TRAIN.ix[row].Weekly_Sales]

    
X_input = np.array(X_input)
Y_output = np.array(Y_output)
print 'Complete.'

print 'Running Linear Regression...'
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_input,Y_output)




print 'Running Test Data through Model...'
X_test = []   

# Re-Arrange data for propper input
# This can probably be optimized significantly
print 'Running Test Data...'
for row in np.arange(115064):
    store_num = TEST.ix[row].Store
#    print X_Store[store_num]
    store_tpe = stype[STORES[STORES.Store==store_num].Type.values[0]]
#    print X_Type[store_tpe]
    store_dept = TEST.ix[row].Dept
#    print X_Dept[store_dept]
#    print X_Week[TRAIN.ix[row].Date.week]
#    print TRAIN.ix[row].IsHoliday*1.
    X = np.concatenate([X_Store[store_num],
                        X_Type[store_tpe],
                        X_Dept[store_dept],
                        X_Week[TEST.ix[row].Date.week],
                        [TEST.ix[row].IsHoliday*1.]])
    X_test += [X]
    #Y_output += [TRAIN.ix[row].Weekly_Sales]

    
X_test = np.array(X_test)
Y_predict = model.predict(X_test)
print 'Complete.'



