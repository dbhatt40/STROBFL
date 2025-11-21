# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:51:13 2025

@author: Divya
"""
from tensorflow.keras import layers, models
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D
import global_vars as gv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def airquality_model():
	main_input = Input(shape=(gv.DATA_DIM,), name='main_input')
	filters =(64,64,128)
	kernel_size = 3
	dropout = 0.2
	x = main_input
	for i, f in enumerate(filters):
		x = Conv1D(f, kernel_size, padding='causal', activation='relu')(x)
		x = BatchNormalization()(x)
        # light downsampling after early layers
		if i == 1:
			x = layers.MaxPooling1D(pool_size=2)(x)

	x = GlobalAveragePooling1D()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(dropout)(x)
	main_output = Dense(gv.NUM_CLASSES)(x)

	model = models.Model(main_input, main_output)  
	return model