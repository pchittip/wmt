#! /usr/bin/env python
"""
===============================================================================
Walmart - Store Sales Forecasting
http://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
Use historical markdown data to predict store sales 
===============================================================================
"""
#print(__doc__)
#from pylab import * 
#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import pandas as pd
#import myUI
from myUI import get_path
import os


train = pd.DataFrame()
test = pd.DataFrame()
stores = pd.DataFrame()
features = pd.DataFrame()
data_dir = os.getcwd() + "\wmt_learn\data"

datafile_train = data_dir + "\\train.csv"
datafile_test = data_dir + "\\test.csv"
datafile_store = data_dir + "\\stores.csv"
datafile_features = data_dir + "\\features.csv"

#print datafile_train






#####################################################################
try:
    wmt_train=pd.read_csv(datafile_train,
                            parse_dates=['Date']
                            )#,index_col=['Store', 'Dept'])
    print 'Training Data Successfully Imported.'
except IOError:
    print IOError
    print datafile_train
    request = 'Please select the training set data file...'
    train_path=get_path('*.csv',request)
    wmt_train=pd.read_csv(train_path,
                            parse_dates=['Date']
                            )#,index_col=['Store', 'Dept'])
##_______________________________________________________________####    
##################################################################### 
train =  wmt_train     
                      
# load TEST data
#####################################################################
try:
    wmt_test=pd.read_csv(datafile_test,
                            parse_dates=['Date']
                            )#,index_col=['Store', 'Dept'])
    print 'Testing Data Successfully Imported.'
except IOError:
    request = 'Please select the testing set data file...'
    test_path = get_path('*.csv',request)
    wmt_test = pd.read_csv(test_path,
                            parse_dates=['Date']
                            )#,index_col=['Store', 'Dept'])  
##_______________________________________________________________####    
##################################################################### 
test =  wmt_test                                

# open store data
# the index is the store number
#####################################################################
try:
    wmt_stores=pd.read_csv(datafile_store)#,index_col=['Store'])
    print 'Store Data Successfully Imported.'
except IOError:
    request = 'Please select the store data file...'
    store_path=get_path('*.csv',request)
    wmt_stores=pd.read_csv(store_path)    
    print 'Store Data Successfully Imported.'
##_______________________________________________________________####    
##################################################################### 
stores=wmt_stores




# open feature data
#####################################################################
try:
    wmt_features=pd.read_csv(datafile_features)
    print 'Feature Data Successfully Imported.'
except IOError:
    request = 'Please select the feature data file...'
    feature_path=get_path('*.csv',request)
    wmt_features=pd.read_csv(feature_path)    
    print 'Feature Data Successfully Imported.'
##_______________________________________________________________####    
#####################################################################  
features = wmt_features



    
    