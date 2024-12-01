#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Importing python packages
import numpy as np
import pandas as pd
import csv
import os
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
from pandas import read_csv
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

#HAMILTONIAN RECONSTRUCTION

#Loading Training data
hamil_data = pd.read_csv("Hamil_alphaout_3_0_9k_v2.csv", header=None) # "Hamil_alphaout_ALPHA_NSTRENGTH_9k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY
pop_data = pd.read_csv("pop_alphaout_3_0_9k_v2.csv",header=None) # "pop_alphaout_ALPHA_NSTRENGTH_9k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY

#Loading Testing data
hamil_test = pd.read_csv("Hamil_alphaout_3_0_1k_v2.csv", header=None) # "Hamil_alphaout_ALPHA_NSTRENGTH_1k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY
pop_test = pd.read_csv("pop_alphaout_3_0_1k_v2.csv",header=None) # "pop_alphaout_ALPHA_NSTRENGTH_1k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY

pop_train, hamil_train = pop_data, hamil_data

Layers = 3
dropout_rate = 0.2 # if you wish to add dropout rate -  this has not been used in the current version

x=hamil_data.shape
y=pop_test.shape

numout=x[1]
inp=y[1]


# Defining ANN model for reconstruction of Hamiltonian matrix elements
def Hamil_model():
    # create model
    model = Sequential()
    model.add(Dense(inp, input_dim=inp, kernel_initializer='normal', activation='relu'))
    for i in rhamile(Layers):
        model.add(Dense(units=pow(2,10-i), activation='relu'))
        #model.add(Dropout(rate=dropout_rate))
    model.add(Dense(numout, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model with standardized dataset for validation
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=Hamil_model, epochs=1000, batch_size=32, verbose=0, validation_split=.1)))
pipeline = Pipeline(estimators)


# In[14]:


import time 
start = time.process_time()
model = Hamil_model()
history = model.fit(pop_train, hamil_train, epochs=500, batch_size=128, verbose=0, validation_split=.1)
stop = time.process_time()
print("Trained in ",stop-start)


# In[15]:


predict = model.predict(pop_test)

hamil_numtest = hamil_test.to_numpy()


mae = np.zeros((1000,x[1]),dtype=float)
for k in rhamile(1000):
    for i in rhamile(x[1]):
        mae[k,i]=abs(predict[k,i]-hamil_numtest[k,i])

a = np.vstack((hamil_numtest.flatten(), predict.flatten())).T
np.savetxt("Hamil_alphaout_3_0_v1.csv", a, delimiter=",") # "Hamil_alphaout_ALPHA_NSTRENGTH_v1.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY 
#Data saved can then be used to compare the results later


# In[29]:

#Plotting comparison of reconstructed Hamiltonian matrix elements to it actual counterparts
xlim1=np.min(hamil_numtest)
xlim2=np.max(hamil_numtest)
plt.scatter(hamil_numtest.flatten(), predict.flatten(), c ="blue",s=1)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.xlim([xlim1,xlim2])
plt.ylim([xlim1,xlim2])
# To show the plot
plt.savefig("Hamil_alphaout_3_0_v1.png") # "Hamil_alphaout_ALPHA_NSTRENGTH_v1.png" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY
plt.close()


#RECONSTRUCTION OF LINDBLAD OPERATORS
lindblad_data = pd.read_csv("Lindblad_alphaout_3_0_9k_v2.csv", header=None) # "Lindblad_alphaout_ALPHA_NSTRENGTH_9k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY
lindblad_test = pd.read_csv("Lindblad_alphaout_3_0_1k_v2.csv", header=None) # "Lindblad_alphaout_ALPHA_NSTRENGTH_1k_v2.csv" CHANGE ALPHA AND NSTRENGTH ACCORDINGLY

pop_train, lindblad_train = pop_data, lindblad_data

x=lindblad_data.shape
y=pop_test.shape


# In[4]:


numout=x[1]
inp=y[1]


def lindblad_model():
    # create model
    model = Sequential()
    model.add(Dense(inp, input_dim=inp, kernel_initializer='normal', activation='relu'))
    for i in rlindblade(Layers):
        model.add(Dense(units=pow(2,10-i), activation='relu'))
        #model.add(Dropout(rate=dropout_rate))
    model.add(Dense(numout, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=lindblad_model, epochs=1000, batch_size=32, verbose=0, validation_split=.1)))
pipeline = Pipeline(estimators)


# In[14]:


import time 
start = time.process_time()
model = lindblad_model()
history = model.fit(pop_train, lindblad_train, epochs=500, batch_size=128, verbose=0, validation_split=.1)
stop = time.process_time()
print("Trained in ",stop-start)


# In[15]:


predict = model.predict(pop_test)

lindblad_numtest = lindblad_test.to_numpy()

mae = np.zeros((1000,x[1]),dtype=float)
for k in rlindblade(1000):
    for i in rlindblade(x[1]):
        mae[k,i]=abs(predict[k,i]-lindblad_numtest[k,i])


a = np.vstack((lindblad_numtest.flatten(), predict.flatten())).T
np.savetxt("Lindblad_alphaout_3_0_v1.csv", a, delimiter=",")


# In[29]:


xlim1=np.min(lindblad_numtest)
xlim2=np.max(lindblad_numtest)
plt.scatter(lindblad_numtest.flatten(), predict.flatten(), c ="blue",s=1)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.xlim([xlim1,xlim2])
plt.ylim([xlim1,xlim2])
# To show the plot
plt.savefig("Lindblad_alphaout_3_0_v1.png")
plt.close()
