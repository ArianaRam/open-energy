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

#%% [1] Unpickle your dataset:

with open('Fluclean', 'rb') as f: 
    Hfull= pickle.load(f)
    
#%% [2] Open file containing customer information:

masterdir= 'D:/users/ariana.ramos/OneDrive - Vlerick Business School/Open Data/Fluvius/real per household electricity/master-table-meters.csv'

# read file, notice change in encoding
master= pd.read_csv(masterdir, header=0, sep=';', decimal= ',', encoding='latin-1')

#get rid of extra columns: 
master= master.drop(columns= ['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'])

#%% [3] keep only the houses in our clean set: 
    
masterC= pd.concat([master.loc[master['InstallatieID']==Hfull.columns[n], ['InstallatieID','Type register', 'Lokale productie']] for n in range(len(Hfull.T))])
masterC=masterC.reset_index() 

#%% [4] Master count how many houses in each tariff
#using reduced set 

tarh, hcnt= np.unique(masterC['Type register'], return_counts=True)

plt.bar(tarh, hcnt)
plt.title('Houses per Tariff Type Full Set')
plt.ylabel('Number of houses')
plt.xlabel('Tariff type')


#%% [5] Master count how many houses have local production 
local, lnum= np.unique(masterC['Lokale productie'], return_counts=True)

plt.bar(local, lnum)
plt.title('Houses with Local Production')
plt.ylabel('Number of Houses')
#plt.xlabel('yes/no')


#%% [6] Summarize house data into one avg day

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

#%% [7] Select each house and average each time period 

Summary= [np.mean((sel(Hfull, n)), axis=0) for n in range(len(Hfull.T))]


#%% [8] Create Array for classification algorithm. 

X= [Summary, [masterC.T.iloc[2][n] for n in range(len(masterC.T.iloc[2]))], [masterC.T.iloc[1][n] for n in range(len(masterC.T.iloc[1]))], [Hfull.columns[n] for n in range(len(Hfull.columns))]]

# we do this to change the structure of the input file so that we can add 'labels'

#this nested list comprehension is adding tariff labels from masterC, house labels from MasterC, 
#and checking that labels match our initial Hfull dataset 

#%% [9] Split dataset into test and train sets 

X_train, X_test, y_train, y_test= train_test_split(X[0], X[1], test_size=0.4, stratify= X[1], random_state= 1)

#use 'stratify' to split into train and test sets that have the same proportion of examples of our target variable. 
#Rows are assigned to the train and test sets randomly. Use random_state=num to always obtain the same random samples.
#This step will divide the sets into two groups 51 and 85 observations each.

#%% [10] Tip: Another way to count samples in the data set is using 'Counter'

print(Counter(y_train))
print(Counter(y_test))

#%% [11] Create your classifier

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# an accuracy score close to 1 means a better performing algorithm 

#%% [12] examine the importance of each predictor: 
importance= mod_dt.feature_importances_

plt.plot(importance)
#this metric indicates where the tree is splitting 

#%% [13] plot decision tree:
feature_names=Hfull.index
    
plt.figure(figsize=(20,15))
plot_tree(mod_dt, feature_names= Hfull.index, class_names=tarh, filled=True)

#Interpretation of Gini index 
# 0 means that all elements belong to a certain class
# 1 means that all elements are randomly distributed across various classes
# 0.5 means equally distributed elements into some classes 

#%% [14] plot outcome 
matplotlib.rc('font', size=14)
matplotlib.rc('axes', titlesize=15, labelsize= 14)
matplotlib.rc('ytick', labelsize= 14)
matplotlib.rc('xtick', labelsize= 14)


fig, ax =plt.subplots(figsize = (15,10))

disp= metrics.plot_confusion_matrix(mod_dt, X_test, y_test, 
                                    display_labels= tarh, 
                                    cmap= plt.cm.Blues,
                                    normalize=None, 
                                    ax=ax )                        
disp.ax_.set_title('Decision Tree Confusion matrix')

#in a perfect prediction all number values would be on the diagonal starting at the top left. 

#%% [15] Try other classifier methods

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

.
