#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[3]:


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


# In[4]:


##create the keras model (LR in this case)
def create_keras_model():
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    ##build LR model
    number_of_classes = 1
    number_of_features = X_train_norm.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(number_of_classes,activation = 'sigmoid',input_dim = number_of_features))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


# In[12]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-ep','--epochs', default = '20')
    parser.add_argument('-ts','--test')
    parser.add_argument('-en','--ensemble')
    
    args = parser.parse_args()
    
    client = args.client
    epochs = args.epochs
    test = args.test
    ensemble = args.ensemble
    
    path = '/gpfs/commons/groups/gursoy_lab/aelhussein/DCI_FL/'
    epochs = int(epochs)
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(f'{path}{client}/Data/')
    # Load and run the neural network
    client_model = tf.keras.models.load_model(f'{path}{client}/model/LR')
    if test == 'False':
        # Evaluate on the validation set    
        validate_loss, validate_auc = client_model.evaluate(X_val_norm, y_val, verbose=0)
        # Train model
        history = tf.keras.callbacks.History()
        with contextlib.redirect_stdout(None):
            client_model.fit(X_train_norm, y_train, verbose=0, epochs=epochs, callbacks=[history])

        # Save model
        client_model.save(f'{path}{client}/model/LR')
        # Calculate loss + auc
        keys = list(history.history.keys())
        train_loss = history.history[keys[0]][-1]
        train_auc = history.history[keys[1]][-1]


        # Send the weights back to master server
        command = f'cp -r {path}{client}/model/LR/* {path}server/model/client_models/{client}/LR'
        subprocess.call(command, shell = True) 

        # Reset stdout and print
        print(len(X_train_norm), train_loss, train_auc, validate_loss, validate_auc)
    
    elif ensemble == 'True':
        client_model = tf.keras.models.load_model(f'{path}server/model/server_models/best_model/LR')
        predicted = client_model.predict(X_test_norm, verbose=0)
        print(len(X_test_norm), predicted.reshape(1,-1))
    
    else:
        client_model = tf.keras.models.load_model(f'{path}server/model/server_models/best_model/LR')
        test_loss, test_auc = client_model.evaluate(X_test_norm, y_test, verbose=0)
        print(len(X_test_norm), 0, 0, test_loss, test_auc)
        
if __name__ == '__main__':
    main()

