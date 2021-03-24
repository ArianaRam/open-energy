# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
import pickle
from matplotlib import cm
import glob
import dill
import matplotlib.dates as mdates
import matplotlib.units as munits


#%%[1] Read all datafile names and generate strings
## Generate string names for filenames


files=[]
path = "D:/users/ariana.ramos/OneDrive - Vlerick Business School/learning algorithms/Datalondon/Power-Networks-LCL-June2015withAcornGps/data"
#substitute path for path in your own computer, note the use of the forward slash(/) 

files = [f for f in glob.glob(path + "**/*.csv", recursive=True)]
#
#%%[2] Read all files (168 names in total): 

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f,header=0, parse_dates = ['DateTime'],usecols=['LCLid', 'DateTime', 'KWH/hh (per half hour) '], na_values=np.nan) for f in files])
#this line may take a few minutes 

#%%[3] Select only 2013

start_date= '2013-01-01 00:00:00'
end_date= '2013-12-31 23:30:00'
mask = (combined_csv['DateTime'] >= start_date) & (combined_csv['DateTime'] <= end_date)

Big2013= combined_csv.loc[mask]

#%%[4] Add python idenfier for missing values

Big2013.iloc[:,2].replace('Null', np.nan, inplace=True)

#%%[5] 
B2= np.array(Big2013)
#

B2=pd.to_numeric(Big2013["KWH/hh (per half hour) "])	
#
#%%
Big2013.insert(2,'KWH/hh', B2)

#Big2013= Big2013.drop(columns='KWH/hh (per half hour) ')
#

#%%[6] Change values column to float
Big2013['KWHhh']= (np.array(Big2013.iloc[:,2]).astype(np.float64)


#%%[7] Drop object column 
Big2013= Big2013.drop(columns='KWH/hh (per half hour) ')

#%%[8] Change to horizontal format

Bighouse= Big2013.reset_index().pivot_table(index='DateTime', columns= 'LCLid', values= 'KWHhh')


#%%[7] Save python variable for analysis
#pick a name for your file

with open('LondondataALL', 'wb') as f: 
    pickle.dump(Big2013,f)
    
#make sure you have enough space on disk, 2.63 GB