#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import argparse
import concurrent.futures
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
import spur
import subprocess
import sys
import tensorflow as tf
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[61]:


def run_clients(path):
    models = ['LR', 'RF', 'SVM']
    ##Run script
    results = {}
    for model in models:
        command = f'python {path}server/centralServer-{model}.py'
        command = command.split(' ')
        output = subprocess.check_output(command) 
        results[model] = output.decode('utf-8').split('\n')[0:2]
    return results


# In[ ]:


def run_ensemble(path):
    clients = ['client_1', 'client_2']
    ensemble_auc = []
    sample_size = []
    for client in clients:
        command = f'python {path}{client}/clientServer-Ensemble.py --client={client}'
        command = command.split(' ')
        output = subprocess.check_output(command)
        output = [float(x) for x in output.decode('utf-8').split(',')]
        ensemble_auc.append(output[1])
        sample_size.append(output[0])
    return np.array(sample_size), np.array(ensemble_auc)


# In[60]:


def averageResults(sample_size, ensemble_auc):
    relative_weights = sample_size / sum(sample_size)
    return np.sum(relative_weights * ensemble_auc)


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s0','--centralServer', default = '')
    parser.add_argument('-s1','--clientServer_1', default = '')
    parser.add_argument('-s2','--clientServer_2', default = '')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clientServer_1 = args.clientServer_1
    clientServer_2 = args.clientServer_2
    

    path = '/gpfs/commons/groups/gursoy_lab/aelhussein/DCI_FL/'
    results = run_clients(path)
    sample_size, ensemble_auc = run_ensemble(path)
    auc = averageResults(sample_size, ensemble_auc)
    print(f'Ensemble AUC: {auc}')

if __name__ == '__main__':
    main()
    

