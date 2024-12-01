#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np

import pandas as pd

import os

import csv

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib.image import NonUniformImage
import matplotlib.gridspec as gridspec
import matplotlib as mp
from matplotlib import cm

from sklearn.model_selection import cross_val_predict

#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier


# In[2]:


from pandas import read_csv
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# In[3]:
dpmax=31

with open('confusion_matrix_KNN_all.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    header = ['Depth','time','count','pop 1-1','1-2','1-3','1-4','2-1','pop 2-2','2-3','2-4','3-1','3-2','pop 3-3','3-4','4-1','4-2','4-3','pop 4-4']
    writer.writerow(header)
    for k in range(2,dpmax):

        pos_data = pd.read_csv("mpg_num_train_class_all_10k.csv", header=None) #in simulation I had taken random numbers for dtheta from -pi/2 to \pi/2
        pop_data = pd.read_csv("mpg_pop_train_class_all_10k.csv",header=None) # have to divide by 5 (due to 5 atoms) it gets summed when sampling J(0)


        # In[4]:


        #importing data for ring trimer
        FF = 0
        pos_test = pd.read_csv("mpg_num_test_class_all_1k.csv", header=None) #in simulation I had taken random numbers for dtheta from -pi/2 to \pi/2
        pop_test = pd.read_csv("mpg_pop_test_class_all_1k.csv",header=None) # have to divide by 5 (due to 5 atoms) it gets summed when sampling J(0)


        # In[5]:


        pop_train, ang_train = pop_data, pos_data




        import time 
        start = time.process_time()
        #clf = RandomForestClassifier(max_depth=k, random_state=0)
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(pop_train, np.ravel(ang_train))
        stop = time.process_time()
        tottime=stop-start


        # In[17]:


        predicted = clf.predict(pop_test)




        ang_numtest = pos_test.to_numpy()


        count = 0
        listindex = []
        for i in range(1200):
            if (predicted[i]==ang_numtest[i]):
                listindex.append(i)
                count += 1
        print(count)



        confusion_matrix(ang_numtest, predicted)


        # In[38]:


        cfff = confusion_matrix(ang_numtest, predicted).ravel()


        # In[ ]:

        row = [k,tottime,count,cfff[0],cfff[1],cfff[2],cfff[3],cfff[4],cfff[5],cfff[6],cfff[7],cfff[8],cfff[9],cfff[10],cfff[11],cfff[12],cfff[13],cfff[14],cfff[15]]
        writer.writerow(row)


    

