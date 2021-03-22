# -*- coding: utf-8 -*-
"""
@author: ariana.ramos
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
import multiprocessing as mp
import pickle
import glob
import gc
from matplotlib import cm
from sklearn.cluster import KMeans
from itertools import groupby
import matplotlib.ticker as mtick 

#set graph styles
sns.set(style="darkgrid")
sns.set_context("paper")

#%% [1]  Read in your dataframe


with open('FluData', 'rb') as f: 
    house= pickle.load(f)
    
#%% [2] Check for missing values in dataframe 

# use the .isna() function

Nantest= house.isna()

# examine the result True/False 

#%% [3]  Add results per house 

Nan= house.isna().sum()

#view total for the set
Nantotal= sum(Nan)

#%% [4] Visualize missing values 

#sort values from big to small
Sort= sorted(Nan, reverse=True)

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(Sort)
ax.set(ylabel="Number of Nans")
ax.set(xlabel="House")
plt.xticks(rotation=45)
ax.set(title="Missing Values Per house FLU")

#%% [5] Calculate percentage 

Nanperc= (pd.DataFrame(Sort)/len(house))


fig, ax =plt.subplots(figsize = (10,4))
ax.plot(Nanperc)
ax.set(ylabel="Percentage of Nans")
ax.set(xlabel="House")
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set(title="Missing Values Per house FLU")

#%% [6]  Decide on a method to fill nans 

housefill= house.fillna(axis=0, method='bfill', limit=8)

# bfill method fills data backwards using the next available value
# limit= 8 defines the amount of consecutive values that will be filled using this method. 
# can you search waht other methods for filling exist? 

#%% Count Nans again 
Nanleft= sum(housefill.isna().sum())  

# if it would not be zero we would have to decide whether to fill more values or drop houses. 

#%%  [7] Observe if dataset has too many consecutive zeros 

zeros0= sorted((housefill==0).sum(axis=0), reverse=True)

perc0= pd.DataFrame(zeros0)/len(house)

#%% [8] Review question 

# Notice the different ways to sum a dataset, or sum over an axis. 

#%%[9] Visualize zeroes 

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(zeros0)
ax.set(ylabel="Number of Zeroes in set")
ax.set(xlabel="House")
plt.xticks(rotation=45)
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set(title="Number of zeroes per house")

#%% [10] Drop houses with too many zeroes

#find houses with less than 3000 zeroes through list comprehension 
mask= [housefill.iloc[:,i] for i in range(len(housefill.T)) if zeros0[i] <= 3000]

#select values in mask in main dataframe to create our full houses dataframe
# transpose to view date in index and houses in columns. 

Hfull= pd.DataFrame(mask).T

#%% [11] Create a function to select one house prior to visualization

def sel(X, n):
    """
    This function:
        1. selects one house in the dataframe
        2. Transforms it to horizontal format (date as rows, time as columns)
        X= input DataFrame
        n= house number to select in dataframe
    """
    h= pd.DataFrame(X.T.iloc[n])
    pivot= pd.pivot_table(h, index= X.index.date, columns=X.index.time)  
    pivot.columns= pivot.columns.get_level_values(1) # drop double indexing in columns
    return pivot 

##TIP: you can save the function in a separate file in your project folder
#  to be able to call it from any script in that same folder. 

#%% View full year 

plt.plot(Hfull.T.iloc[8])
plt.title('Energy Consumption over one year')
plt.xlabel('Month')
plt.ylabel('Energy [kWh]')

#TIP try it with different houses and see what preliminary conclusions you can make. 


#%% [13] Reshape house to view heatmap

#select the house you want to examine
H0= sel(Hfull,8)

#What shape is your DataFrame now? 

#%% [14] Visualize house days

plt.plot(H0.T.values)
plt.title('Energy consumption')
plt.xlabel('Time period')
plt.ylabel('Energy[kWh]')

# this graph shows an overlay of the daily consumption curves 

#%% [15] Visualize as a heatmap

ax= sns.heatmap(H0, cmap= 'YlGnBu', yticklabels=20, xticklabels=6) 
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

#%% Limit plot to exclude high outlier values 

ax=sns.heatmap(H0, cmap= 'YlGnBu', yticklabels=20, xticklabels=6, vmin=0, vmax=0.2)
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

#%% [13] Review question 
# Can you make a function to create a graph or heatmap? 

#%% [14] Visualize the whole set aggregated

totdem= pd.DataFrame(np.sum(Hfull.T), index= Hfull.index)
aggdem= pd.pivot_table(totdem, index=totdem.index.date, columns= totdem.index.time)
aggdem.columns= aggdem.columns.get_level_values(1)

#%% [15] Visualize aggregated data

##Year view 

plt.plot(totdem)
plt.title('Aggregated Energy Consumption over one year')
plt.xlabel('Month')
plt.ylabel('Energy [kWh]')

#%% Day view 

plt.plot(aggdem.T.values)
plt.title('Energy consumption')
plt.xlabel('Time period')
plt.ylabel('Energy[kWh]')

#%% Heatmap 

ax=sns.heatmap(aggdem, cmap= 'YlGnBu', yticklabels=20, xticklabels=6, vmin=0, vmax=15)
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

# What trends can you observe in the data? 

#%% Pickle your clean dataset

with open('Fluclean', 'wb') as f: 
    pickle.dump(Hfull, f)

