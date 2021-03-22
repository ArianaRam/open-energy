# -*- coding: utf-8 -*-
"""
@author: ariana.ramos
"""
## this is where I can write my script
### This cell imports the necessary modules and sets a few plotting parameters for display
#we can write anything here
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
import pickle

#seaborn is a plotting package
sns.set(style="darkgrid")
sns.set_context("paper")




#%% [1] Reading data from a single file

### Read in the data
# Shift + Enter, or press the play button above 

hname= 'D:/users/ariana.ramos/OneDrive - Vlerick Business School/Open Data/Fluvius/real per household electricity/READING_2016/READING_2016.CSV'

FLU2016= pd.read_csv(hname, header=0,sep=';', decimal= ',' , parse_dates= [3])

# the data is read into a pandas 'DataFrame' structure. 

#%% [3]


### The .head() function shows the first few lines of data for perspecitve
FLU2016.head()

#%% [4] 

### Lists column names 

FLU2016.columns
#%% [5] 

#The .shape function shows the shape of your dataframe

shape= FLU2016.shape

#%%  [6] 

## REVIEW QUESTIONS: 
# How many rows and columns are in 'FLU2016'? 
# What is the name of the column that contains the energy measurements?

#%% [9] Rename power measurements column, offtake/intake column, and time date column

FLU2016 = FLU2016.rename(columns= {'Afname/Injectie': 'Offtake/Injection', 'Meetwaarde': 'Energy', 'Meter read tijdstip': 'DateTime'})

# check column names again 
FLU2016.columns

#%% [8]  Take a first look at the data


plt.plot(FLU2016['Energy'])
plt.title('First look at input data')
plt.ylabel(ylabel='kWh')
plt.xlabel(xlabel= '???')

#%% [9] What about injection and offtake?

#Mark injection into the network as negative (eg. from a PV Panel)
FLU2016.loc[FLU2016['Offtake/Injection']=='Injectie', 'Energy']= FLU2016['Energy'] *-1


#%% [10] take a look at data again 

plt.plot(FLU2016['Energy'])
plt.title('Second look at input data')
plt.ylabel(ylabel='kWh')
plt.xlabel(xlabel= '???')

#%% [11] Pivot per consumer ID 

house= pd.pivot_table(FLU2016, values='Energy', index= 'DateTime', columns='InstallatieID')

# sort the index (row labels) in the correct order 
house= house.sort_index()
#sorting not working because 'datetime is not well recognized'



#%% [12] 

# Review question: 
    # examine the shape of the 'house' dataframe
    # How many houses are there? 
    # How many time periods are there? 
#%% [13] convert index to time date 
#01JAN16:00:00:00
dates= pd.to_datetime(house.index, format='%d%b%y:%H:%M:%S')

#%% [14] 
house.index= dates

house= house.sort_index()



#%% [15]  # select one house and take a closer look 

# the .iloc function is used to select a column or a slice
# see the first house on the set
house1= house.T.iloc[0]

#%% [16]

myFmt = DateFormatter('%m')
fig, ax= plt.subplots(figsize= (10,6))
ax.plot(house1)
ax.set(title='House 0 Energy Consupmtion 2016')
ax.set(xlabel= 'Month', ylabel= 'Energy [kWh]')
ax.xaxis.set_major_formatter(myFmt); 
ax.tick_params(axis='x', labelrotation=45)
 # month formatter 

#%%[17] # Review question 

# can you find a house with PV production? 

#%% [18] # Save your DataFrame 

with open('FluData', 'wb') as f: 
    pickle.dump(house, f)
