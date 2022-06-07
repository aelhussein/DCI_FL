#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
import _pickle as cPickle
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[16]:


##create the keras model (LR in this case)
def create_RF_model(path):
    model = RandomForestClassifier(max_depth=8, random_state=0)
    with open(f'{path}server/model/server_models/current_model/RF/RF', 'wb') as f:
        cPickle.dump(model, f)
    return model


# In[32]:


##load the rf  model
def save_RF_model(location, client_model):
    with open(location, 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[3]:


##load the rf  model
def load_RF_model(location):
    with open(location, 'rb') as f:
        return  cPickle.load(f)


# In[4]:


def clear_clients(client, path):
    ##clear models from clients
    command = f'rm -rf {path}{client}/model/RF/*'
    command.split(' ')
    subprocess.call(command, shell = True)
    return


# In[34]:


def run_clients(client, path, dir, test):
    ##send model to clients
    command = f'cp -r {dir}/model/server_models/current_model/RF/* {path}{client}/model/RF/'
    subprocess.call(command, shell = True) 
    
    ##Run script
    command = f'python {path}{client}/clientServer-RF.py -cl={client} -ts={test}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[6]:


def fedAvg(server_1_model, server_2_model, relative_weights):
    ##extract individual trees
    client_trees = [server_1_model.estimators_, server_2_model.estimators_]
    ##assign number of trees
    num_trees = [round(len(client_trees[0])*weight) for weight in relative_weights]
    ##sample from both forests
    new_forest = []
    for i in range(len(client_trees)):
        new_forest.extend(list(np.random.choice(client_trees[i], num_trees[i])))
    ##assign to models
    server_1_model.estimators_ = new_forest
    server_2_model.estimators_ = new_forest
    return server_1_model, server_2_model


# In[7]:


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
    parser.add_argument('-s1','--clientServer_1', default = '')
    parser.add_argument('-s2','--clientServer_2', default = '')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clientServer_1 = args.clientServer_1
    clientServer_2 = args.clientServer_2
    
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
    model = create_RF_model(path)
    # Set runtime parameters
    test = False
    # Run model
    with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(run_clients, client_1, path, dir, test)
        command_2 = executor.submit(run_clients, client_2, path, dir, test)

        # Retrieve server responses
        server_1_response = [float(j) for j in command_1.result()]
        server_2_response = [float(j) for j in command_2.result()]

    # Wait until the two servers return their weights files
    while (os.path.isfile(path + 'server/model/client_models/client_1/RF/RF') == False or
               os.path.isfile(path + 'server/model/client_models/client_2/RF/RF' == False)):
        time.sleep(5)

    # Calculate weights and AUC
    relative_weights, current_auc = validateResults(server_1_response, server_2_response)
    print(f'Validation AUC: {current_auc}')

    # Load models
    server_1_model = load_RF_model(f'{path}server/model/client_models/client_1/RF/RF')
    server_2_model = load_RF_model(f'{path}server/model/client_models/client_2/RF/RF')

    # Conduct federated averaging to update the federated_model
    server_1_model, server_2_model = fedAvg(server_1_model, server_2_model, relative_weights)
    save_RF_model(f'{path}server/model/server_models/current_model/RF/RF', server_1_model)
    # Save the model in best_model for ensemble learning
    save_RF_model(f'{path}server/model/server_models/best_model/RF/RF', server_1_model)

    test = True

    if test:
         with concurrent.futures.ThreadPoolExecutor() as executor:
            command_1 = executor.submit(run_clients, client_1, path, dir, test)
            command_2 = executor.submit(run_clients, client_2, path, dir, test)

            # Retrieve server responses
            server_1_response = [float(j) for j in command_1.result()]
            server_2_response = [float(j) for j in command_2.result()]

            relative_weights, current_auc = validateResults(server_1_response, server_2_response)

            print(f'Test AUC: {current_auc}')

if __name__ == '__main__':
    main()

