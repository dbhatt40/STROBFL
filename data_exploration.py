# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:14:37 2023

@author: Divya
"""

import numpy as np
data = np.load("C:\\Users\\Divya\\OneDrive\\Documents\\OU-NewResearch\\ModelPoisoningCode\\trunk\\global_weights_t0.npy", allow_pickle=True)
print("*******************\n")
print("Global data shape", data.shape)
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)
print(data[3].shape)
print(data[4].shape)
print(data[5].shape)
print(data[6].shape)
print(data[7].shape)
print("*******************\n")
data = np.load("C:\\Users\\Divya\\OneDrive\\Documents\\OU-NewResearch\\ModelPoisoningCode\\trunk\\ben_delta_0_t0.npy", allow_pickle=True)
print("*******************\n")
print("Ben data shape", data.shape)
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)
print(data[3].shape)
print("*******************\n")
data = np.load("C:\\Users\\Divya\\OneDrive\\Documents\\OU-NewResearch\\ModelPoisoningCode\\trunk\\ben_delta_sample1.npy", allow_pickle=True)
print("Ben data sample shape", data.shape)
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)
print(data[3].shape)
print('test')