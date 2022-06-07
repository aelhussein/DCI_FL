#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")
import contextlib
import math
import pickle as pkl
import random
import spur
import sys
import argparse
import subprocess
from sklearn import metrics


# In[83]:


##load the data and create train/test split
def dataloader(path):
    ##load features and cohort data
    X = pd.read_csv(path+'X.csv', index_col = 'hadm_id')
    y = pd.read_csv(path+'y.csv', index_col = 'hadm_id')

    ## train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state=1)
    #create validation set too
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    ## create scaler and apply only to numeric data before adding binary data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.iloc[:,:-5])
    X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index, columns = X_train.columns[:-5])
    X_train_norm = X_train_norm.merge(X_train.iloc[:,-5:], left_index = True, right_index = True)

    ##apply scaler to test data
    X_test_norm = scaler.transform(X_test.iloc[:,:-5])
    X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index, columns = X_test.columns[:-5])
    X_test_norm = X_test_norm.merge(X_test.iloc[:,-5:], left_index = True, right_index = True)
    
    ##apply scaler to val data
    X_val_norm = scaler.transform(X_val.iloc[:,:-5])
    X_val_norm = pd.DataFrame(X_val_norm, index = X_val.index, columns = X_val.columns[:-5])
    X_val_norm = X_val_norm.merge(X_val.iloc[:,-5:], left_index = True, right_index = True)
    
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[91]:


def test_probability_clients(path, client):
    models = ['LR', 'RF', 'SVM']
    results = {}
    for model in models:
        command = f'python {path}{client}/clientServer-{model}.py -cl={client} -ts=Pass -en=True'
        command = command.split(' ')
        output = subprocess.check_output(command) 
        output =  output.decode('utf-8').split('\n')
        results[model]= prob_parser(output)
    return results


# In[85]:


def prob_parser(output):
    parsed = []
    for i in output:
        parsed.extend([float(x) for x in i.replace('[', '').replace(']','').split(' ') if x != ''])
    sample_size = parsed[0]
    preds = parsed[1:]
    return sample_size, preds


# In[86]:


def ensemble_learner(results):
    ensemble = []
    for _, preds in results.values():
        ensemble.append(preds)
    ensemble = np.array(ensemble)
    ensemble_mean = np.mean(ensemble, axis = 0)
    return ensemble_mean


# In[ ]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    
    args = parser.parse_args()
    
    client = args.client
    path = '/gpfs/commons/groups/gursoy_lab/aelhussein/DCI_FL/'
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(f'{path}{client}/Data/')
    
    results = test_probability_clients(path, client)
    ensemble_mean = ensemble_learner(results)
    fpr, tpr, thresholds = metrics.roc_curve(y_test.values, ensemble_mean)
    ensemble_auc = metrics.auc(fpr, tpr)
    print(f'{y_test.shape[0]}, {ensemble_auc}')

if __name__ == '__main__':
    main()

