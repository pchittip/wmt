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

stype = {'A': 0, 'B': 1, 'C': 2}


#X_store = np.zeros(45)
#X_type = np.zeros(3)
#X_size = np.zeros(1)
#X_dept = np.zeros(98)
#X_week = np.zeros(52)
#X_holiday = np.zeros(1)

A=np.zeros(45)
X=np.identity(45)
X_Store = np.vstack((A, X[X[:,0] < 45]))

X_Type = np.identity(3)

A=np.zeros(98)
X=np.identity(98)
X_Dept = np.vstack((A, X[X[:,0] < 98]))

X_size = np.zeros(1)

A=np.zeros(53)
X=np.identity(53)
X_Week = np.vstack((A, X[X[:,0] < 53]))

TRAIN = wmt_learn.train

#for row in TRAIN.index:
#    print TRAIN.ix[row]
X_input = []   
Y_output = [] 

print 'Pre-Processing Data...'
for row in np.arange(1000):
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





