# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:17:07 2025
@author: Divya

"""
#########################
# Purpose: Utility functions for uci sensor data with concept drift
########################

import pandas as pd
import os
from keras.models import Model
from keras.layers import Input, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.utils import np_utils

import global_vars as gv

def data_uci_sensor():
	
    data_path =  "/content/STROBFL/data/gas_sensor/gas_drift_all_batches.csv"
# 	data_path =  "/content/STROBFL/data/gas_sensor/batchesCSV/batch1.csv"
  
    df = pd.read_csv(data_path)
    df = df.replace(r'\b\d+:\s*', '', regex=True)
    print("UCI Sensor Dataset shape:", df.shape)
    print(df.head())
    X = df.iloc[1:, 1:gv.DATA_DIM+1].to_numpy()
    y = df.iloc[1:,0].to_numpy()
	
    print("UCI Sensor x,y shape:", X.shape, y.shape)
	
    split_point = int(len(X) * 0.2)
    X_test, X_train = X[:split_point], X[split_point:]
    y_test, y_train = y[:split_point], y[split_point:]

#     X_train, X_test, y_train, y_test = train_test_split(
# 	    X, y, test_size=0.2, shuffle='false')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
	
	
    if y_train.min() == 1:
		   y_train = y_train - 1
    if y_test.min() == 1:
           y_test = y_test - 1

    y_train = np_utils.to_categorical(y_train, gv.NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, gv.NUM_CLASSES)
	
    return X_train, y_train, X_test, y_test

def uci_sensor_model():
 	main_input = Input(shape=(gv.DATA_DIM,), name='main_input')
 	x = Dense(256, use_bias=True, activation='relu')(main_input)
 	x = Dropout(0.5)(x)
 	x = Dense(256, use_bias=True, activation='relu')(x)
 	x = Dropout(0.5)(x)
 	# main_output = Dense(1)(x)
 	main_output = Dense(gv.NUM_CLASSES)(x)
 	model = Model(inputs=main_input, outputs=main_output)
 	return model





