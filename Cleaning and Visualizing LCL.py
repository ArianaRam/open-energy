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

#%%  [1] Input variable 

with open('LdALLBIGhouse', 'rb') as a: 
    LCLall= pickle.load(a)

#%% [2] Check for missing values in dataframe 

Nan= LCLall.isna().sum()


mask= [LCLall.iloc[:, i] for i in range(len(LCLall.T)) if Nan[i]<=500]

house= pd.DataFrame(mask).T
Nantotal= sum(Nan)

#%%[3] Visualize missing values 


#sort values from big to small
Sort= sorted(Nan, reverse=True)

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(Sort)
ax.set(ylabel="Number of Nans")
ax.set(xlabel="House")
plt.xticks(rotation=45)
ax.set(title="Missing Values Per house LCL")

#%% [4] Calculate Percentage 
Nanperc= (pd.DataFrame(Sort)/len(LCLall))

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(Nanperc)
ax.set(ylabel="Percentage of Nans")
ax.set(xlabel="House")
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set(title="Missing Values Per house LCL")


#%% [5] Create a mask to select Nan threshhold value


mask= [LCLall.iloc[:, i] for i in range(len(LCLall.T)) if Nan[i]<=0]

house= pd.DataFrame(mask).T

#%% [6] Check missing values again 
Nans2= pd.DataFrame(sorted(house.isna().sum(), reverse=True))

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(Nans2/len(Hfull))
ax.set(ylabel="Percentage of Nans")
ax.set(xlabel="House")
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set(title="Missing Values Per house LCL")

#%% [7] Decide on a method to fill nans 

housefill= house.fillna(axis=0, method='bfill')

#%%[8] Count Nans that are left 
Nanleft= sum(housefill.isna().sum())

#%% [9] Use ffill to fill remaining missing values
housefill2= house.fillna(axis=0, method= 'ffill')

#%%[10] Check for zeroes 
zeros0= sorted((housefill2==0).sum(axis=0), reverse=True) 
perc0= pd.DataFrame(zeros0)/len(house)

#%%[11] Visualize zeroes 

fig, ax =plt.subplots(figsize = (10,4))
ax.plot(zeros0)
ax.set(ylabel="Number of Zeroes in set")
ax.set(xlabel="House")
plt.xticks(rotation=45)
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set(title="Number of zeroes per house")

#%% [12] Drop houses with too many zeroes 
# find houses with less than 1500 zeroes through list comprehension

mask2= [housefill2.iloc[:,i] for i in range(len(housefill2.T)) if zeros0[i] <= 1500]

Hfull= pd.DataFrame(mask2).T

##THIS IS MY CLEAN DATASET FOR ANALYSIS 


#%%[13] #%% View full year
sns.set(style="darkgrid")
sns.set_context("paper")

plt.plot(Hfull)
#plt.ylim([0,10])
plt.title('Clean LCL Dataset')
plt.xlabel('Month')
plt.ylabel('Energy [kWh/hh]') 

sns.set(style="dark")
sns.set_context("paper")

#might take a few minutes to run 

#%%[14] View one house

plt.plot(Hfull.T.iloc[8])
plt.title('Energy Consumption over one year')
plt.xlabel('Month')
plt.ylabel('Energy [kWh/hh]')

#%%[15]Create a function to select one house prior to visualization

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


#%%[16] Reshape to view heatmap

#select the house you want to examine
H0= sel(Hfull,8)

#%% [17] Visualize house days 

plt.plot(H0.T.values)
plt.title('Energy consumption')
plt.xlabel('Time period')
plt.ylabel('Energy[kWh]/hh')

#%% [18] Visualize as heatmap


ax= sns.heatmap(H0, cmap= 'YlGnBu', yticklabels=20, xticklabels=6) 
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

#%% [19] Limit plot to exclude high outlier values 

ax=sns.heatmap(H0, cmap= 'YlGnBu', yticklabels=20, xticklabels=6, vmin=0, vmax=0.2)
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

#%% [20] Visualize Aggregated set 
totdem= pd.DataFrame(np.sum(Hfull.T), index= Hfull.index)
aggdem= pd.pivot_table(totdem, index=totdem.index.date, columns= totdem.index.time)
aggdem.columns= aggdem.columns.get_level_values(1)

#%% [21] Visualize aggregated data 
# year view 

plt.plot(totdem)
plt.title('Aggregated Energy Consumption over one year')
plt.xlabel('Month')
plt.ylabel('Energy [kWh/hh]')

#%% [22] Day view


plt.plot(aggdem.T.values)
#plt.ylim([0,16])
plt.title('Aggregated Energy consumption')
plt.xlabel('Time period')
plt.ylabel('Energy[kWh/hh]')


#%% [23] Heatmap


ax=sns.heatmap(aggdem, cmap= 'YlGnBu', yticklabels=20, xticklabels=6, vmin=0)
plt.title('Input data heatmap')
plt.xlabel('Time Period')
plt.ylabel('Date')

# What trends can you observe in the data? 

#%%[24] Picle your clean dataset 

with open('LCLclean', 'wb') as f: 
    pickle.dump(Hfull, f)