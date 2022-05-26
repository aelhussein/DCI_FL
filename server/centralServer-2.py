#!/usr/bin/env python
# coding: utf-8

# In[101]:


import warnings
warnings.filterwarnings("ignore")
import argparse
import concurrent.futures
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import shutil
import spur
import subprocess
import sys
import tensorflow as tf
import time
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[27]:


##create the keras model (LR in this case)
def create_keras_model():
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    ##build LR model
    number_of_classes = 1
    number_of_features = 27
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(number_of_classes,activation = 'sigmoid',input_dim = number_of_features))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


# In[114]:


def clear_clients(client, path):
    ##clear models from clients
    command = f'rm -rf {path}{client}/model/*'
    command.split(' ')
    subprocess.call(command, shell = True)
    return


# In[147]:


def run_clients(client, path, dir, epochs, test):
    ##send model to clients
    if test:
        command = f'cp -r {dir}/model/server_models/best_model/* {path}{client}/model'
        subprocess.call(command, shell = True) 
    else:
        command = f'cp -r {dir}/model/server_models/current_model/* {path}{client}/model'
        subprocess.call(command, shell = True) 
    
    ##Run script
    command = f'python {path}{client}/clientServer-2.py -cl={client} -ep={epochs} -ts={test}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[133]:


def fedAvg(server_1_model, server_2_model, relative_weights):
    weights = [server_1_model.get_weights(), server_2_model.get_weights()]
    new_weights = []
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.array([np.average(np.array(
            weights_), axis=0, weights=relative_weights) for weights_ in zip(*weights_list_tuple)]))
    return new_weights


# In[142]:


def validateResults(server_1_response, server_2_response):
    # Check if the current validation is better than the previous best one
    relative_weights = [server_1_response[0] / (server_1_response[0] + server_2_response[0]),
                        server_2_response[0] / (server_1_response[0] + server_2_response[0])]

    current_validation = server_1_response[3] * relative_weights[0] + server_2_response[3] * relative_weights[1]
    
    current_auc = server_1_response[4] * relative_weights[0] + server_2_response[4] * relative_weights[1]
    
    return relative_weights, current_validation, current_auc


# In[156]:


centralServer = 'pe2cc3-002'
clientServer_1 = 'pe2cc3-002'
clientServer_2 = 'pe2cc3-002'

client_1 = 'client_1'
client_2 = 'client_2'


# Hyperparameters
patience = 2
epochs = 20

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
federated_model = create_keras_model()
federated_model.save(f'{path}server/model/server_models/current_model')
federated_model.save(f'{path}server/model/server_models/best_model')

# Set runtime parameters
patience_counter = 0
iterations = 0
lowest_validation = float('inf')
early_stopping = False
test = False
while (early_stopping == False) and (iterations < 10):
    # Run model
    with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(run_clients, client_1, path, dir, epochs, test)
        command_2 = executor.submit(run_clients, client_2, path, dir, epochs, test)

        # Retrieve server responses
        server_1_response = [float(j) for j in command_1.result()]
        server_2_response = [float(j) for j in command_2.result()]
    # Wait until the two servers return their weights files
    while (os.path.isfile(path + 'server/model/client_models/client_1/saved_model.pb') == False or
               os.path.isfile(path + 'server/model/client_models/client_2/saved_model.pb' == False)):
        time.sleep(5)

    relative_weights, current_validation, current_auc = validateResults(server_1_response, server_2_response)
    if current_validation < lowest_validation:
        patience_counter = 0
        federated_model.save(f'{path}server/model/server_models/best_model')
        lowest_validation = current_validation
        print(f'Validation Loss: {current_validation} Validation AUC: {current_auc}')
    else:
        patience_counter += 1

    # Conduct federated averaging to update the federated_model if we have not exceeded patience
    if patience_counter > patience:
        early_stopping = True
        test = True
    else:
        server_1_model = tf.keras.models.load_model(path + 'server/model/client_models/client_1/')
        server_2_model = tf.keras.models.load_model(path + 'server/model/client_models/client_2/')
        new_weights = fedAvg(server_1_model, server_2_model, relative_weights)
        federated_model.set_weights(new_weights)
        federated_model.save(f'{path}server/model/server_models/current_model')
    iterations +=1
    
    if iterations >= 10:
        test = True

if test:
     with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(run_clients, client_1, path, dir, epochs, test)
        command_2 = executor.submit(run_clients, client_2, path, dir, epochs, test)

        # Retrieve server responses
        server_1_response = [float(j) for j in command_1.result()]
        server_2_response = [float(j) for j in command_2.result()]
        
        relative_weights, current_validation, current_auc = validateResults(server_1_response, server_2_response)
        
        print(f'Test AUC: {current_auc}')


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


# Hyperparameters
patience = 2
epochs = 20

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
federated_model = create_keras_model()
federated_model.save(f'{path}server/model/server_models/current_model')
federated_model.save(f'{path}server/model/server_models/best_model')

# Set runtime parameters
patience_counter = 0
iterations = 0
lowest_validation = float('inf')
early_stopping = False
test = False
while (early_stopping == False) and (iterations < 10):
    # Run model
    with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(run_clients, client_1, path, dir, epochs, test)
        command_2 = executor.submit(run_clients, client_2, path, dir, epochs, test)

        # Retrieve server responses
        server_1_response = [float(j) for j in command_1.result()]
        server_2_response = [float(j) for j in command_2.result()]
    # Wait until the two servers return their weights files
    while (os.path.isfile(path + 'server/model/client_models/client_1/saved_model.pb') == False or
               os.path.isfile(path + 'server/model/client_models/client_2/saved_model.pb' == False)):
        time.sleep(5)

    relative_weights, current_validation, current_auc = validateResults(server_1_response, server_2_response)
    if current_validation < lowest_validation:
        patience_counter = 0
        federated_model.save(f'{path}server/model/server_models/best_model')
        lowest_validation = current_validation
        print(f'Validation Loss: {current_validation} Validation AUC: {current_auc}')
    else:
        patience_counter += 1

    # Conduct federated averaging to update the federated_model if we have not exceeded patience
    if patience_counter > patience:
        early_stopping = True
        test = True
    else:
        server_1_model = tf.keras.models.load_model(path + 'server/model/client_models/client_1/')
        server_2_model = tf.keras.models.load_model(path + 'server/model/client_models/client_2/')
        new_weights = fedAvg(server_1_model, server_2_model, relative_weights)
        federated_model.set_weights(new_weights)
        federated_model.save(f'{path}server/model/server_models/current_model')
    iterations +=1
    
    if iterations >= 10:
        test = True

if test:
     with concurrent.futures.ThreadPoolExecutor() as executor:
        command_1 = executor.submit(run_clients, client_1, path, dir, epochs, test)
        command_2 = executor.submit(run_clients, client_2, path, dir, epochs, test)

        # Retrieve server responses
        server_1_response = [float(j) for j in command_1.result()]
        server_2_response = [float(j) for j in command_2.result()]
        
        relative_weights, current_validation, current_auc = validateResults(server_1_response, server_2_response)
        
        print(f'Test AUC: {current_auc}')

if __name__ == '__main__':
    main()

