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


#reload(wmt_learn)

#raw_input('Hit any key when ready')


#wmt_learn.load()
import os
#print os.path.realpath(__file__)
#print os.path.dirname(__file__)
#print os.path._getfullpathname(__file__)

## start by looking at the different store types
wmt_learn.stores.sort_index(by=['Size'],inplace=True,ascending=False)


plt.close('all')
fig = plt.figure()
#fig
ax1 = fig.add_subplot(111)
store_sizes=wmt_learn.stores['Size']

wmt_learn.stores.Store.values

bars = ax1.bar(range(1,46), store_sizes, color='blue', edgecolor='black')

#pull out the A stores and make them red
Astores = wmt_learn.stores[wmt_learn.stores['Type']=='A']
for store_num in Astores['Store']:
    bars[store_num-1].set_facecolor('red')


#pull out the B stores and make them green
Bstores = wmt_learn.stores[wmt_learn.stores['Type']=='B']
for store_num in Bstores['Store']:
    bars[store_num-1].set_facecolor('green')

    
    
#plt.xticks(range(1,45))
plt.show()