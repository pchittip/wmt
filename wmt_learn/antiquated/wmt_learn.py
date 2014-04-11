#! /usr/bin/env python
"""
===============================================================================
Walmart - Store Sales Forecasting
http://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
Use historical markdown data to predict store sales 
===============================================================================
"""
print(__doc__)
from pylab import * 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from myUI import get_path


def load():
    print 'loaded'
    
        
    plt.cla()
    plt.clf()
    plt.close('all')
    
    # Make the graphs a bit prettier, and bigger
    pd.set_option('display.mpl_style', 'default') 
    pd.set_option('display.line_width', 5000) 
    pd.set_option('display.max_columns', 60) 
    
    #figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    #plt.rcParams['figure.figsize'] = (15, 5)
    #plt.figure(figsize=(15,5))
    #plt.rc(figure.figsize=15,5)
    datafile_train = 'train.csv'
    datafile_test = 'test.csv'
    datafile_store = 'stores.csv'
    datafile_features = 'features.csv'
    verbose=True
    
    
    # references
    # http://pandas.pydata.org/pandas-docs/dev/io.html#io-read-csv-table
    # http://pandas.pydata.org/pandas-docs/stable/tutorials.html
    
    # notes
    # the parse_dates option in pd.read_csv changes the string to a python date
    
    
    print 'Importing data from *.csv to Pandas DataFrames...\n'
    
    
    
    
    
    
    
    # open TRAIN data
    #####################################################################
    try:
        wmt_train=pd.read_csv(datafile_train,
                                parse_dates=['Date']
                                )#,index_col=['Store', 'Dept'])
        print 'Training Data Successfully Imported.'
    except IOError:
        request = 'Please select the training set data file...'
        train_path=get_path('*.csv',request)
        wmt_train=pd.read_csv(train_path,
                                parse_dates=['Date']
                                )#,index_col=['Store', 'Dept'])
    ##_______________________________________________________________####    
    #####################################################################    
    
    
            
                    
                                    
    # load TEST data
    #####################################################################
    try:
        wmt_test=pd.read_csv(datafile_test,
                                parse_dates=['Date']
                                )#,index_col=['Store', 'Dept'])
        print 'Testing Data Successfully Imported.'
    except IOError:
        request = 'Please select the testing set data file...'
        test_path=get_path('*.csv',request)
        wmt_test=pd.read_csv(test_path,
                                parse_dates=['Date']
                                )#,index_col=['Store', 'Dept'])  
    ##_______________________________________________________________####    
    ##################################################################### 
    
        
            
                
                    
                            
    
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
        
    
    
    print '\nraw data loaded: (wmt_train, wmt_test, wmt_stores, wmt_features).'
    print '-'*80
    #
    #
    #
    #
    #
    #
    #if(verbose):  
    #    print ' '
    #    print '    wmt_train = ' + str(wmt_train.columns.get_values())
    #    print '    wmt_test = ' + str(wmt_test.columns.get_values())
    #    print '    wmt_stores = ' + str(wmt_stores.columns.get_values())
    #    print '    wmt_features = \n     ' + str(wmt_features.columns.get_values())
    #    print ' '
    #    print '    Time Span of Training Data:'
    #    print '    ' + wmt_train.Date.min().date().isoformat()
    #    print '    ' + wmt_train.Date.max().date().isoformat()
    #    td=(wmt_train.Date.max().date()-wmt_train.Date.min().date()).days
    #    tdw = td/7
    #    tdy = round(td/365.,2)
    #    print '    ' + str(tdw) + ' weeks (' + str(tdy) +' years)'
    #
    #
    #    print '\nThe Pandas DataFrames use a 2D index (Store, Dept)'
    #    print 'This gives us a unique key for each data entry.\n'
    #    
    #    print 'Syntax'
    #    print '-------'
    #    print 'Get all the data from Store #22:'
    #    print "    wmt_train.ix[22]"""
    #    print ' '
    #    
    #    print 'Get the data from Store #22 that pertains to department #14:'
    #    print "    wmt_train.ix[22].ix[14]"""
    #    print ' '
    #    
    #    print 'Get entire column(s) of data across all stores and departments:'
    #    print "    wmt_train[['Date', 'Weekly_Sales']]"
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ## notes
    ## groupby is a way to split apply combine data
    ## http://youtu.be/w26x-z-BdWQ?t=40m56s
    #
    ###testing
    ##get the sales data for store 22
    ##store_22_weeklySales = wmt_train.ix[22][['Date','Weekly_Sales']]
    ### group the data by date (because there will be duplicate dates across depts)
    ###store_22_weeklySales=store_22_weeklySales.groupby('Date', sort=True)
    ##store_22_weeklySales.sort_index(by=['Date'],inplace=True)
    ##store_22_grouped=store_22_weeklySales.groupby(['Date'])
    ##store_22_totalSales=store_22_grouped.sum()
    ##store_22_totalSales.plot()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #wmt_train[['Date', 'Weekly_Sales']]
    #
    #grouped_train = wmt_train.groupby(level='Store', sort=True)
    
    #dive into training set
    wmt_train.columns.get_values()
    array(['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'], dtype=object)
    
    #Sort
    wmt_train.sort_index(by=['Store','Date','Dept'],inplace=True)
    
    # Investigate Store #1
    stID=1
    store_01_train = wmt_train[wmt_train.Store == stID]
    store_01_info = wmt_stores[wmt_stores.Store == stID]
    store_01_featuresbyWeek = wmt_features[wmt_features.Store == stID]
    
    
    
    grp = store_01_train.groupby(['Date'])
    sales_nums = grp['Weekly_Sales'].sum()
    sales_nums.plot()
    print '\ndone.'
    print '-'*80
    
    
    from numpy.random import rand
    
    Z = rand(10,10)
    
    plt.figure()
    c = plt.pcolor(Z)
    plt.title('default: no edges')
    
    
    c = plt.pcolor(Z, edgecolors='k', linewidths=2)
    plt.xlim((0,10))
    
    plt.title('2010-02-05 (Sales by Dept)')
    
    d = store_01_train[store_01_train.Date == '2010-02-05'][['Dept','Weekly_Sales']]
    
    
    Z=zeros((100,1))
    
    d['Dept'].values
    
    d['Weekly_Sales'].values
    
    for idex in arange(len(d['Dept'].values)):
        Z[ d['Dept'].values[idex]-1 ] = d['Weekly_Sales'].values[idex]
    
    #Z=Z/Z.max()
    Z=Z.reshape((10,10))
    Z=Z.T
    c = plt.pcolor(Z, edgecolors='k', linewidths=2,cmap='Reds')
    plt.xlim((0,10))
    plt.yticks([])
    
    plt.xticks([])
    
    plt.annotate('01',xy=(3,1),xytext=(.5,.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('02',xy=(3,1),xytext=(.5,1.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('03',xy=(3,1),xytext=(.5,2.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('04',xy=(3,1),xytext=(.5,3.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('05',xy=(3,1),xytext=(.5,4.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('06',xy=(3,1),xytext=(.5,5.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('07',xy=(3,1),xytext=(.5,6.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('08',xy=(3,1),xytext=(.5,7.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('09',xy=(3,1),xytext=(.5,8.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('10',xy=(3,1),xytext=(.5,9.5),horizontalalignment='center',verticalalignment='center')
    
    
    plt.annotate('11',xy=(3,1),xytext=(1.5,.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('12',xy=(3,1),xytext=(1.5,1.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('13',xy=(3,1),xytext=(1.5,2.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('14',xy=(3,1),xytext=(1.5,3.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('15',xy=(3,1),xytext=(1.5,4.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('16',xy=(3,1),xytext=(1.5,5.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('17',xy=(3,1),xytext=(1.5,6.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('18',xy=(3,1),xytext=(1.5,7.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('19',xy=(3,1),xytext=(1.5,8.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('20',xy=(3,1),xytext=(1.5,9.5),horizontalalignment='center',verticalalignment='center')
    
    plt.annotate('21',xy=(3,1),xytext=(2.5,.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('22',xy=(3,1),xytext=(2.5,1.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('23',xy=(3,1),xytext=(2.5,2.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('24',xy=(3,1),xytext=(2.5,3.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('25',xy=(3,1),xytext=(2.5,4.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('26',xy=(3,1),xytext=(2.5,5.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('27',xy=(3,1),xytext=(2.5,6.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('28',xy=(3,1),xytext=(2.5,7.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('29',xy=(3,1),xytext=(2.5,8.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('30',xy=(3,1),xytext=(2.5,9.5),horizontalalignment='center',verticalalignment='center')
    
    
    plt.annotate('31',xy=(3,1),xytext=(3.5,.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('32',xy=(3,1),xytext=(3.5,1.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('33',xy=(3,1),xytext=(3.5,2.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('34',xy=(3,1),xytext=(3.5,3.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('35',xy=(3,1),xytext=(3.5,4.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('36',xy=(3,1),xytext=(3.5,5.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('37',xy=(3,1),xytext=(3.5,6.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('38',xy=(3,1),xytext=(3.5,7.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('39',xy=(3,1),xytext=(3.5,8.5),horizontalalignment='center',verticalalignment='center')
    plt.annotate('40',xy=(3,1),xytext=(3.5,9.5),horizontalalignment='center',verticalalignment='center')
    
    plt.figure()
    plt.plot(d['Dept'].values,d['Weekly_Sales'].values)
    
    plt.show()