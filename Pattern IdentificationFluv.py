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

#%% [1] Unpickle your dataset

with open('Fluclean', 'rb') as f: 
    Hfull= pickle.load(f)


#%% [2] Pattern identification through K-means 

# Finding Representative consumers 
# n_clusters= need to decide the number of clusters in advance 

kmeans= KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y= kmeans.fit_predict(Hfull.T)    #here you input your clean dataset
centers= kmeans.cluster_centers_       #these will be the representative patterns
labs= kmeans.labels_                   
inertia= kmeans.inertia_               #sum of squared distances of samples to their closest cluster center. 
n_iter=kmeans.n_iter_                  # allows you to check how many iterations the algorithm does before converging. 


#%% [3] format the results

Pattern = pd.DataFrame(centers.T)
Pattern.index= Hfull.index

#%% [4] Visualize your results 

# Full year view
plt.plot(Pattern)
plt.title('Energy Consumption Patterns 2016')
plt.xlabel('Month')
plt.ylabel('Energy [kWh]')

#might be slow to run 
#what can you tell about the different patterns? 

#%% [5] Select time period function

# write a function to select a time period. 
def Period(X,date1,date2):
    mask= (X.index >= date1) & (X.index <= date2)
    Xsub= X.loc[mask]
    return Xsub


#%% [6] Select a time period
date1= '2016-03-05 00:00:00'
date2= '2016-03-08 23:30:00'

March5= Period(Pattern, date1, date2)

#%% [7]  Visualize time period 


myFmt = DateFormatter('%H:%M') 
fig, ax = plt.subplots(figsize = (12,4))
ax.plot(March5.iloc[:,0], '-o') 
#select each pattern individually to add identifyer in legend eg. '-o'
ax.plot(March5.iloc[:,1], '-s')
ax.plot(March5.iloc[:,2], '-+')
ax.plot(March5.iloc[:,3], '--')
ax.plot(March5.iloc[:,4], '-D')
ax.legend(['K1', 'K2', 'K3', 'K4', 'K5'])
ax.set(xlabel="Hour", ylabel="kWh")
ax.set(title="Energy Consumption per Cluster");
ax.xaxis.set_major_formatter(myFmt); 
plt.savefig('clustM5flu.png', format= 'png', dpi= 600)	

# TIP: play around with date1 and date2 to see the differences between different dates, try out longer periods of time. 


#%% [8] Visualize heatmaps 

#Use the 'sel' function we created in our previous file 'Cleaning and Visualizing' to select one pattern. 

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


P1= sel(Pattern,0)
P2= sel(Pattern,1)
P3= sel(Pattern,2)
P4= sel(Pattern,3)
P5= sel(Pattern,4)

#%% [9] Compare two patterns 

f, (ax1, ax2) = plt.subplots(1,2, figsize= (15,6),  gridspec_kw={'width_ratios':[1,1]})
f.suptitle('Fluvius Representative Consumer Clusters', size=14)

g1= sns.heatmap(P1, cmap= 'YlGnBu', ax= ax1, cbar= False, vmin=-1, vmax=2, yticklabels=20, xticklabels=6)
#change vmin and vmax to select value range
# change yticklabels and xticklabels to select spacing between y and x ticks. 
# don't change anything else
ax1.set_title('Pattern 1')
ax1.set_ylabel('Date')
ax1.set_xlabel('Hour')
g2= sns.heatmap(P4, cmap= 'YlGnBu', ax= ax2, vmin=-1, vmax=2, cbar= True, yticklabels=20, xticklabels=6)
ax2.set_title('Pattern 4')
ax2.set_xlabel('Hour')
ax2.set_yticks([])


for ax in [g1,g2]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
    
# TIP: compare other patterns. 

#%% [10] Sensitivity analysis 
#Within Clusters Sum of Squares 


def elbow(X): 
    wcss= []
    Cents=[]
    Labels=[]
    n_iter=[]
#WCSS: within cluster sum of squares
    for i in range(1,20):
        kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit_predict(X.T)
        wcss.append(kmeans.inertia_)
        Cents.append(kmeans.cluster_centers_)
        Labels.append(kmeans.labels_)
        n_iter.append(kmeans.n_iter_)
   
    return wcss, Cents, Labels, n_iter

#%% [11] Run your elbow function

wcss, cents, labels, n_iter= elbow(Hfull)

#%% [12] Visualize elbow

fig, ax= plt.subplots(figsize=(12,4))
plt.plot(wcss)
ax.set(xlabel='Number of Clusters', ylabel= 'WCSS')
ax.set(title='Cluster Dispersion Indicator')
ax.set_xticks(np.arange(1,20))
ax.set(xlim=(-0,18))


#%% [13] Select one instance of sensitivity analysis

Sens3= pd.DataFrame(cents[3])
Sens3.columns= Hfull.index

# Can you visualize the results with different amounts of clusters? 


#%% [14]

with open('FLUpatterns', 'wb') as f: 
    pickle.dump([P1, P2, P3, P4, P5], f)

with open('FLUsensitivity', 'wb') as f: 
   pickle.dump([wcss, cents, labels, n_iter], f)

