#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
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
import time
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import _pickle as cPickle
sns.set()


# In[43]:


def create_SVM_model(path):
    ##0 weight vector for number of features
    model = SGDClassifier(loss = 'hinge', penalty = 'l2', max_iter =1)
    model = CalibratedClassifierCV(model)
    ##save in central
    with open(f'{path}server/model/server_models/current_model/SVM/SVM.pkl', 'wb') as f:
        cPickle.dump(model, f)


# In[4]:


##load the rf  model
def save_SVM_model(path, client_model):
    with open(f'{path}server/model/server_models/current_model/SVM/SVM.pkl', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[5]:


def load_SVM_model(location):
    with open(location, 'rb') as f:
        return  cPickle.load(f)


# In[6]:


def clear_clients(client, path):
    ##clear models from clients
    command = f'rm -rf {path}{client}/model/SVM/*'
    command.split(' ')
    subprocess.call(command, shell = True)
    return


# In[7]:


def run_clients(client, path, dir, test, RBF):
    ##send model to clients
    command = f'cp -r {dir}/model/server_models/current_model/SVM/* {path}{client}/model/SVM/'
    subprocess.call(command, shell = True) 
    
    ##Run script
    command = f'python {path}{client}/clientServer-SVM.py -cl={client} -ts={test} -rb={RBF}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[22]:


def exctract_coefs(client_model):
    coef_avg = 0
    for m in client_model.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
    coef_avg  = coef_avg/len(client_model.calibrated_classifiers_)
    return coef_avg


# In[20]:


def fedAvg(server_1_model, server_2_model, relative_weights):
    #extract weights
    server_1_coefs, server_2_coefs = exctract_coefs(server_1_model), exctract_coefs(server_1_model)
    #average weights
    weights = [server_1_coefs, server_2_coefs]
    weights_ = np.average(weights, weights = relative_weights, axis = 0)
    # assign to the classifier
    for m in server_1_model.calibrated_classifiers_:
        m.base_estimator.coef_ = weights_
    for m in server_2_model.calibrated_classifiers_:
        m.base_estimator.coef_ = weights_
    return server_1_model, server_2_model


# In[21]:


def validateResults(server_1_response, server_2_response):
    # Check if the current validation is better than the previous best one
    relative_weights = [server_1_response[0] / (server_1_response[0] + server_2_response[0]),
                        server_2_response[0] / (server_1_response[0] + server_2_response[0])]

    current_auc = server_1_response[2] * relative_weights[0] + server_2_response[2] * relative_weights[1]

    
    return relative_weights, current_auc


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s0','--centralServer', default = '')
    parser.add_argument('-s1','--clientServer_1', default = 'client_1')
    parser.add_argument('-s2','--clientServer_2', default = 'client_2')
    parser.add_argument('-rb','--RBF', default = 'False')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clientServer_1 = args.clientServer_1
    clientServer_2 = args.clientServer_2
    RBF = args.RBF
    
    client_1 = 'client_1'
    client_2 = 'client_2'

      ##load directory
    __file__ = 'centralServer.ipynb'
    dir = os.path.abspath(os.path.dirname(__file__))
    path = '/gpfs/commons/groups/gursoy_lab/aelhussein/DCI_FL/'

    # Delete past client data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(
            clear_clients, client_1, path)
        command_2 = executor.submit(
            clear_clients, client_2, path)

    # Model architecture
    model = create_SVM_model(path)
    # Set runtime parameters
    test = False
    lowest_auc = float('inf')
    early_stopping = False
    iterations = 0
    patience_counter = 0
    patience = 2
    RBF = False
    # Run model
    while (early_stopping == False) and (iterations < 10):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            command_1 = executor.submit(run_clients, client_1, path, dir, test, RBF)
            command_2 = executor.submit(run_clients, client_2, path, dir, test, RBF)

            # Retrieve server responses
            server_1_response = [float(j) for j in command_1.result()]
            server_2_response = [float(j) for j in command_2.result()]

        # Wait until the two servers return their weights files
        while (os.path.isfile(path + 'server/model/client_models/client_1/SVM/SVM.pkl') == False or
                   os.path.isfile(path + 'server/model/client_models/client_2/SVM/SVM.pkl' == False)):
            time.sleep(5)

        # Calculate weights and AUC
        relative_weights, current_auc = validateResults(server_1_response, server_2_response)
        print(f'Validation AUC: {current_auc}')


        # Load models
        server_1_model= load_SVM_model(f'{path}server/model/client_models/client_1/SVM/SVM.pkl')
        server_2_model= load_SVM_model(f'{path}server/model/client_models/client_2/SVM/SVM.pkl')

        # Conduct federated averaging to update the federated_model
        server_1_model, server_2_model  = fedAvg(server_1_model, server_2_model, relative_weights)

        save_SVM_model(path, server_1_model)

        if current_auc < lowest_auc:
            patience_counter = 0
            with open(f'{path}server/model/server_models/best_model/SVM/SVM.pkl', 'wb') as f:
                    cPickle.dump(server_1_model, f)
            lowest_f1 = current_auc
            print(f'Best validation AUC: {current_auc}')
        else:
            patience_counter += 1

        iterations +=1

        if (patience_counter > patience) or (iterations >= 10):
            early_stopping = True
            test = True


    if test:
         with concurrent.futures.ThreadPoolExecutor() as executor:
            command_1 = executor.submit(run_clients, client_1, path, dir, test, RBF)
            command_2 = executor.submit(run_clients, client_2, path, dir, test, RBF)

            # Retrieve server responses
            server_1_response = [float(j) for j in command_1.result()]
            server_2_response = [float(j) for j in command_2.result()]

            relative_weights, current_auc = validateResults(server_1_response, server_2_response)

            print(f'Test AUC: {current_auc}')
   
   
            
if __name__ == '__main__':
    main()

