# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:01:38 2021

@author: ariana.ramos
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:39:58 2021

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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot

#seaborn is a plotting package
sns.set(style="darkgrid")
sns.set_context("paper")

#%%[1] Clean dataset
with open('LCLclean', 'rb') as f:
    LCLHfull= pickle.load(f)  


#%%[2] Load houshold information file 
f= 'informations_households.csv'

master= pd.read_csv(f, header=0, sep=',')

#%%[3] Keep only houses in our clean set: 
masterC= pd.concat([master.loc[master['LCLid']==LCLHfull.columns[n], ['LCLid', 'stdorToU','Acorn', 'Acorn_grouped']] for n in range(len(LCLHfull.T))])
masterC=masterC.reset_index() 

# remove acorns with few values
m3= masterC.loc[(masterC['Acorn_grouped']!= 'ACORN-U')]

LCL2= LCLHfull[[m3['LCLid'].iloc[n] for n in range(len(m3))]]
    
#%%[4] Visualize acorn groups
Acorngroup= masterC.groupby(['Acorn_grouped']).count()['LCLid']

#bar plot

plt.bar(range(len(Acorngroup)), Acorngroup)
plt.xticks(ticks= range(len(Acorngroup)), labels=Acorngroup.index, rotation=45)
plt.ylabel('Number of houses')
plt.xlabel('Acorn group')
plt.title('Number of houses per Acorn')

#%% [5] Visualize tariff type
Tartyp= masterC.groupby(['stdorToU']).count()['LCLid']

#bar plot

plt.bar(range(len(Tartyp)), Tartyp)
plt.xticks(ticks= range(len(Tartyp)), labels= Tartyp.index, rotation= 45)
plt.ylabel('Number of Houses')
plt.xlabel('Tariff Type')
plt.title('Number of houses per tariff type')


#%% [6] Visualize stacked bars
actar= pd.DataFrame(masterC.groupby(['Acorn_grouped', 'stdorToU']).count()['LCLid'])

Stack=pd.pivot_table(actar, index= 'stdorToU', columns= 'Acorn_grouped', values= 'LCLid')

Stack.plot.bar(stacked=True)


#%% [7] Summarize house data into one avg day

#function to select each house in the dataframe 
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

#%% [8] Make summary

Summary= [np.mean((sel(LCL2, n)), axis=0) for n in range(len(LCL2.T))]

#might take a few minutes to run 


#%% [9] Create Array for classification algorithm. 


X= [Summary, [m3.iloc[n,1] for n in range(len(m3))], [m3.iloc[n,0] for n in range(len(m3))]]

#%% [10] split into train test sets


X_train, X_test, y_train, y_test= train_test_split(X[0], X[1], test_size=0.4, stratify= X[1], random_state= 1)

#%% [11] Create classifier 

#We will work with a classification and regression tree (CART) algorithm

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))


#%% [12] examine the importance of each predictor: 
importance= mod_dt.feature_importances_

plt.plot(importance)

#%% [13] plot decision tree:

    
plt.figure(figsize=(20,15))
plot_tree(mod_dt, feature_names= LCLHfull.index, class_names=Tartyp.index, filled=True)


#%% [14] plot outcome 
matplotlib.rc('font', size=14)
matplotlib.rc('axes', titlesize=15, labelsize= 14)
matplotlib.rc('ytick', labelsize= 14)
matplotlib.rc('xtick', labelsize= 14)


fig, ax =plt.subplots(figsize = (15,10))

disp= metrics.plot_confusion_matrix(mod_dt, X_test, y_test, 
                                    display_labels= Tartyp.index, 
                                    cmap= plt.cm.Blues,
                                    normalize=None, 
                                    ax=ax )                        
disp.ax_.set_title('Decision Tree Confusion matrix')

#in a perfect prediction all number values would be on the diagonal starting at the top left. 

#%% [15] Try other classifiers 

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=6, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X[0], X[1], cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#kfold means that it will select different sets of X for training and testing

# LR   - logistic regresstion 
# LDA  - linear discriminant analysis
# KNN  - K nearest neighbors classifier
# CART - classification and regression tree
# NB   - Gaussian naive bayes
# SVM  - Support vector machine 


#%% [16] Compare algorithms
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.ylabel('Accuracy Score')
pyplot.show()


