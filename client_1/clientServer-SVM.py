#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")
import contextlib
import math
import _pickle as cPickle
import random
import sys
import argparse
import subprocess


# In[3]:


##load the data and create train/test split
def dataloader(path, RBF):
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
    
    ##turn into arrays
    x = X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = tuple(map(np.array, x))
    
    ##RBF if transform wanted
    if RBF == 'True':
        rbf_sampler = RBFSampler(gamma =1, random_state =1)
        X_train_norm = rbf_sampler.fit_transform(X_train_norm)
        X_val_norm = rbf_sampler.transform(X_val_norm)
        X_test_norm = rbf_sampler.transform(X_test_norm)
    
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[4]:


##load the rf  model
def load_SVM_model(path, client):
    with open(f'{path}{client}/model/SVM/SVM.pkl', 'rb') as f:
        return  cPickle.load(f)


# In[5]:


##load the rf  model
def save_SVM_model(path, client, client_model):
    with open(f'{path}{client}/model/SVM/SVM.pkl', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[24]:


def exctract_coefs(client_model):
    coef_avg = 0
    for m in client_model.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
    coef_avg  = coef_avg/len(client_model.calibrated_classifiers_)
    return coef_avg


# In[27]:


def fedAvg(server_1_model_coef, server_2_model_coef, relative_weights):
    weights = [server_1_model_coef, server_2_model_coef]
    return np.average(weights, weights = relative_weights, axis = 0)


# In[33]:


def exctract_coefs(client_model):
    coef_avg = 0
    for m in client_model.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
    coef_avg  = coef_avg/len(client_model.calibrated_classifiers_)
    return coef_avg


# In[1]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-ts','--test')
    parser.add_argument('-rb','--RBF', default = 'False')
    parser.add_argument('-en','--ensemble')
    
    args = parser.parse_args()
    
    client = args.client
    test = args.test
    RBF = args.RBF
    ensemble = args.ensemble
    path = '/gpfs/commons/groups/gursoy_lab/aelhussein/DCI_FL/'
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(f'{path}{client}/Data/', RBF)
    # Load and run the neural network
    client_model= load_SVM_model(path, client)
    if test == 'False':
        # Fit on the train data
        client_model.fit(X_train_norm, y_train)
        y_train_pred = client_model.predict_proba(X_train_norm)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_pred)
        train_auc = metrics.auc(fpr, tpr)
        # Evaluate on the validation set
        y_val_pred = client_model.predict_proba(X_val_norm)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_pred)
        val_auc = metrics.auc(fpr, tpr)
        # Save model
        save_SVM_model(path, client, client_model)
        # Send the weights back to master server
        command = f'cp -r {path}{client}/model/SVM/* {path}server/model/client_models/{client}/SVM/'
        subprocess.call(command, shell = True) 

        # Reset stdout and print
        print(len(X_train_norm), train_auc, val_auc)

    elif ensemble == 'True':
        with open(f'{path}server/model/server_models/best_model/SVM/SVM.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        client_model = CalibratedClassifierCV(client_model)
        client_model.fit(X_train_norm, y_train)
        prediction = client_model.predict_proba(X_test_norm)[:,1]
        print(len(X_test_norm), prediction) 
        
    else:
        with open(f'{path}server/model/server_models/best_model/SVM/SVM.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        y_test_pred = client_model.predict_proba(X_test_norm)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
        test_auc = metrics.auc(fpr, tpr)
        print(len(X_test_norm),0 , test_auc) 

        
if __name__ == '__main__':
    main()

