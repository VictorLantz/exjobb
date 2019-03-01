#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 

@author: victor
"""

import numpy as np
from numpy import linalg as ln
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras import models, layers
from keras.layers import Dropout, Dense, Input
from keras.utils import multi_gpu_model
from keras import optimizers, metrics, regularizers
from keras.models import load_model, save_model, Model
from sklearn.decomposition import PCA
import random

seed = 7
random.seed(seed)
file_name = 'parameters_' + str(seed) + '.txt'

train_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/pot_train_d.txt'
train_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/eig_train_d.txt'
val_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/pot_val_d.txt'
val_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/eig_val_d.txt'
test_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/pot_test_d.txt'
test_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/eig_test_d.txt'
 
train_pot = np.loadtxt(train_pot_file)
train_eig = np.loadtxt(train_eig_file)
val_pot = np.loadtxt(val_pot_file)
val_eig = np.loadtxt(val_eig_file)
test_pot = np.loadtxt(test_pot_file)
test_eig = np.loadtxt(test_eig_file)

def decoder_transform(encoder):
    pass
 
def encoder_pipeline(input_dim,
                    output_dim,
                    enc_node,
                    act_fun = 'relu',
                    output_fun = 'linear',
                    opt_fun = 'adam',
                    cost_fun = 'mean_squared_error',
                    dropout_rate = 0.0,
                    lr_rate = 0.01,
                    lambd = 0.00):
    
    method = getattr(keras.optimizers, opt_fun)
    
    encoder_input = Input(shape=(input_dim,), dtype='float32', name='encoder_input')
    X = encoder_input
    
    for nodes in enc_node:
        X = Dense(nodes, 
                  activation = act_fun,
                  kernel_regularizer = regularizers.l2(lambd))(X)
        if(dropout_rate != 0):
            X = Dropout(dropout_rate)(X)
    encoder_output = Dense(output_dim, activation = output_fun )(X)
    
    encoder =  Model(inputs=[encoder_input], outputs=[encoder_output])
    encoder.compile(optimizer = method(lr = lr_rate),
                  loss = cost_fun,
                 metrics=['accuracy', 'mse', 'mae'])    
    return encoder

def training_plot(history):
    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    epochs = range(1, len(mse) + 1)
    plt.plot(epochs[2:], mae[2:], 'bo', label='Training mae')
    plt.plot(epochs[2:], val_mae[2:], 'b', label='Validation mae')
    plt.title('Training and validation mean absolute error')
    plt.legend()
    plt.figure()
    plt.plot(epochs[2:], mse[2:], 'bo', label='Training mse')
    plt.plot(epochs[2:], val_mse[2:], 'b', label='Validation mse')
    plt.title('Training and validation mean squared error')
    plt.legend()
    plt.show()

def file_creator(saved_params, saved_rms, saved_mae, saved_models):
    param_writer = open('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/' + file_name, 'w')
    for i in range(len(saved_params)):
        param_writer.write(str(saved_rms[i]) + ' ' + str(saved_mae[i]) + ' ')
        for j in range(len(saved_params[i])):
            param_writer.write(str(saved_params[i,j]) + ' ')
        param_writer.write('\n')
        saved_models[i].save('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/model_'+str(saved_rms[i])+'.h5')
    print('file done')

def parameter_generator(nbr_of_sets, inp_dim, out_dim):
    #0-7 nodes in layers
    parameter_matrix=np.zeros((nbr_of_sets,13))
    for parameter_set in parameter_matrix:
        nbr_layers = random.randint(4,8)
        prev_layer = inp_dim
        for i in range(nbr_layers):
            decrease = random.randint(0, np.floor((prev_layer-out_dim)/2))
            new_layer = prev_layer - decrease
            if new_layer > out_dim:
                parameter_set[i] = new_layer
            else:
                parameter_set[i] = out_dim
            prev_layer = new_layer
            #half = random.randint(0,1)
            #parameter_set[i+6] = 0.1*random.randint(0,4) + 0.5*half
     #8 dropout       
        half = random.randint(0,1)
        parameter_set[8] = 0.1*random.randint(0,4) + 0.05*half
     #9 learning rate   
        lr_exp = random.randint(2,5)
        parameter_set[9] = 10**-lr_exp
     #10 lambda value   
        lambd_exp = random.randint(1,5)
        parameter_set[10] = 10**-lambd_exp
     #11 batch size
        #batch_size = random.randint(0,1)
        parameter_set[11] = 64
     #12 number of epochs   
        epochs = random.randint(40,60)
        parameter_set[12] = epochs
    return parameter_matrix
        
def model_selector(nbr_of_models, nbr_saved, input_dim, output_dim):
    
    parameters = parameter_generator(nbr_of_models, input_dim, output_dim)
    saved_rms = np.ones(nbr_saved)*1000 #just some large number as inital
    saved_mae = np.ones(nbr_saved)*1000
    saved_parameters = np.zeros((nbr_saved, 13))
    saved_models = [0]*nbr_saved
    
    for parameter_set in parameters:
        nodes = []
        for entry in parameter_set[:8]:
            if (entry != 0):
                nodes = nodes + [int(entry)]
        
        model = encoder_pipeline(input_dim,
                                     output_dim,
                                     nodes,
                                     act_fun = 'relu',
                                     output_fun = 'linear',
                                     opt_fun = 'adam',
                                     cost_fun = 'mean_squared_error',
                                     dropout_rate = parameter_set[8],
                                     lr_rate = parameter_set[9],
                                     lambd =  parameter_set[10])
        
        callback_list = [keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error'
                                                           ,factor = 0.1, patience = 5)]
        history = model.fit(train_pot, train_eig, 
                            batch_size = int(parameter_set[11]),
                            epochs = int(parameter_set[12]),
                            validation_data = (val_pot,val_eig),
                            callbacks = callback_list,
                            verbose = 0)
        
        mse_loss = history.history['val_mean_squared_error']
        mae_loss = history.history['val_mean_absolute_error']
        
        min_mse = min(mse_loss)
        min_mae = min(mae_loss)
        min_loss = min_mse
        
        for i in range(len(saved_rms)):
            if (min_loss<saved_rms[i]):
                temp_rms = saved_rms[i]
                temp_mae = saved_mae[i]
                temp_para = saved_parameters[i].copy()
                temp_mod = saved_models[i]
                
                saved_rms[i] = min_mse
                saved_mae[i] = min_mae
                saved_parameters[i] = parameter_set.copy()
                saved_models[i] = model
                
                for j in range(i+1, len(saved_rms)):

                    temp_rms2 = saved_rms[j]
                    temp_mae2 = saved_mae[j]
                    temp_para2 = saved_parameters[j].copy()
                    temp_mod2 = saved_models[j]
                    
                    saved_rms[j] = temp_rms
                    saved_mae[j] = temp_mae
                    saved_parameters[j] = temp_para.copy()
                    saved_models[j] = temp_mod
                    
                    temp_rms = temp_rms2
                    temp_mae = temp_mae2
                    temp_para = temp_para2.copy()
                    temp_mod = temp_mod2

                break
    return saved_parameters, saved_rms, saved_mae, saved_models

    
node_list = [800, 600, 400, 200, 200, 200, 200]
encoder = encoder_pipeline(1001,
                    200,
                    node_list,
                    act_fun = 'relu',
                    output_fun = 'linear',
                    opt_fun = 'adam',
                    cost_fun = 'mean_absolute_error',
                    dropout_rate = 0.0,
                    lr_rate = 0.001,
                    lambd = 0.0001)

encoder.summary()
history = encoder.fit(train_pot,train_eig, batch_size=64, epochs=50,
                    validation_data = (val_pot,val_eig)) 
#encoder.save('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoder2.h5')

#run model selection
#saved_params, saved_rms, saved_mae, saved_models = model_selector(40, 4, 1001, 200)
#file_creator(saved_params, saved_rms, saved_mae, saved_models)
