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

seed = 5
random.seed(seed)
file_name = 'parameters_' + str(seed) + '.txt'

encoder900_5 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/900encoder5.h5'

encoder200_2 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoder2.h5'

encoder200_ny = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoderny.h5'
encoder100_ny = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/100encoderny.h5'
polyencoder_200 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200polyencoderny.h5'

chosen_encoder = polyencoder_200

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

def autoencoder_pipeline(inp_dim,
                    feat_dim,
                    dec_node,
                    enc_model,
                    act_fun = 'relu',
                    output_fun = 'linear',
                    opt_fun = 'adam',
                    cost_fun = 'mean_squared_error',
                    dropout_rate = 0.2,
                    lr_rate = 0.001,
                    lambd = 0.0):
    
    method = getattr(keras.optimizers, opt_fun)
    
    encoder = load_model(enc_model)
    encoder.trainable = False
    
    decoder_input = Input(shape=(feat_dim,), dtype='float32', name='decoder_input')
    X = decoder_input
    for nodes in dec_node:
        X = Dense(nodes, 
                  activation = act_fun,
                  kernel_regularizer = regularizers.l2(lambd))(X)
        if(dropout_rate != 0):
            X = Dropout(dropout_rate)(X)
                
    decoder_output = Dense(inp_dim, activation = output_fun )(X)
    
    decoder =  Model(inputs=[decoder_input], outputs=[decoder_output])
    decoder.compile(optimizer = method(lr = lr_rate),
                  loss = cost_fun,
                 metrics=['accuracy', 'mse', 'mae'])    
    
    autoencoder = keras.models.Model(encoder.inputs, decoder(encoder.output))
    for layer in autoencoder.layers[:-1]:
        layer.trainable= False
    
    autoencoder.compile(optimizer = method(lr = lr_rate),
                  loss = cost_fun,
                  metrics=['accuracy', 'mse', 'mae'])  
    
    return autoencoder

def encoder_pipeline(inp_dim,
                    feat_dim,
                    enc_node,
                    act_fun = 'relu',
                    output_fun = 'linear',
                    opt_fun = 'adam',
                    cost_fun = 'mean_squared_error',
                    dropout_rate = 0.0,
                    lr_rate = 0.01,
                    lambd = 0.00):
    
    method = getattr(keras.optimizers, opt_fun)
    
    encoder_input = Input(shape=(inp_dim,), dtype='float32', name='encoder_input')
    X = encoder_input
    
    for nodes in enc_node:
        X = Dense(nodes, 
                  activation = act_fun,
                  kernel_regularizer = regularizers.l2(lambd))(X)
        if(dropout_rate != 0):
            X = Dropout(dropout_rate)(X)
    encoder_output = Dense(feat_dim, activation = output_fun )(X)
    
    encoder =  Model(inputs=[encoder_input], outputs=[encoder_output])
    encoder.compile(optimizer = method(lr = lr_rate),
                  loss = cost_fun,
                 metrics=['accuracy', 'mse', 'mae'])    
    return encoder

def training_plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    loss = loss[2:]
    val_loss = val_loss[2:]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs[2:], loss, 'bo', label='Training loss')
    plt.plot(epochs[2:], val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def file_creator(saved_params, saved_rms, saved_models):
    param_writer = open('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/' + file_name, 'w')
    for i in range(len(saved_params)):
        param_writer.write(str(saved_rms[i]) + ' ')
        for j in range(len(saved_params[i])):
            param_writer.write(str(saved_params[i,j]) + ' ')
        param_writer.write('\n')
        saved_models[i].save('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/model_'+str(saved_rms[i])+'.h5')
    print('file done')

def parameter_generator(nbr_of_sets, feat_dim, out_dim):
    #0-5 nodes in layers
    parameter_matrix=np.zeros((nbr_of_sets,11))
    for parameter_set in parameter_matrix:
        nbr_layers = random.randint(3,6)
        prev_layer = feat_dim
        for i in range(nbr_layers):
            increase = random.randint(0, np.floor((out_dim-prev_layer)*0.75))
            new_layer = prev_layer + increase
            if new_layer < out_dim:
                parameter_set[i] = new_layer
            else:
                parameter_set[i] = out_dim
            prev_layer = new_layer
            #half = random.randint(0,1)
            #parameter_set[i+6] = 0.1*random.randint(0,4) + 0.5*half
     #6 dropout       
        half = random.randint(0,1)
        parameter_set[6] = 0.1*random.randint(0,4) + 0.05*half
     #7 learning rate   
        lr_exp = random.randint(2,5)
        parameter_set[7] = 10**-lr_exp
     #8 lambda value   
        lambd_exp = random.randint(1,5)
        parameter_set[8] = 10**-lambd_exp
     #9 batch size
        #batch_size = random.randint(0,1)
        batch_size = 1
        if batch_size ==0:
            parameter_set[9] = 8000
        elif(batch_size ==1):
            parameter_set[9] = 64
     #10 number of epochs   
        epochs = random.randint(40,60)
        parameter_set[10] = epochs
    return parameter_matrix
        
def model_selector(nbr_of_models, nbr_saved, feat_dim, output_dim):
    
    parameters = parameter_generator(nbr_of_models, feat_dim, output_dim)
    saved_rms = np.ones(nbr_saved)*1000 #just some large number as inital
    saved_parameters = np.zeros((nbr_saved, 11))
    saved_models = [0]*nbr_saved
    
    for parameter_set in parameters:
        nodes = []
        for entry in parameter_set[:6]:
            if (entry != 0):
                nodes = nodes + [int(entry)]
        model = autoencoder_pipeline(output_dim,
                                     feat_dim,
                                     nodes,
                                     chosen_encoder,
                                     act_fun = 'relu',
                                     output_fun = 'linear',
                                     opt_fun = 'adam',
                                     cost_fun = 'mean_squared_error',
                                     dropout_rate = parameter_set[6],
                                     lr_rate = parameter_set[7],
                                     lambd =  parameter_set[8])
        
        history = model.fit(train_pot, train_pot, 
                            batch_size = int(parameter_set[9]),
                            epochs = int(parameter_set[10]),
                            validation_data = (val_pot,val_pot),
                            verbose = 0)
        
        val_loss = history.history['val_mean_squared_error']
        min_loss = min(val_loss)
        
        for i in range(len(saved_rms)):
            if (min_loss<saved_rms[i]):
                temp_rms = saved_rms[i]
                temp_para = saved_parameters[i].copy()
                temp_mod = saved_models[i]
                
                saved_rms[i] = min_loss
                saved_parameters[i] = parameter_set.copy()
                saved_models[i] = model
                
                for j in range(i+1, len(saved_rms)):

                    temp_rms2 = saved_rms[j]
                    temp_para2 = saved_parameters[j].copy()
                    temp_mod2 = saved_models[j]
                    
                    saved_rms[j] = temp_rms
                    saved_parameters[j] = temp_para.copy()
                    saved_models[j] = temp_mod
                    
                    temp_rms = temp_rms2
                    temp_para = temp_para2.copy()
                    temp_mod = temp_mod2

                break
    return saved_parameters, saved_rms, saved_models

    
#node_list = [600, 400, 200, 200, 200, 200]
#encoder = encoder_pipeline(1001,
#                    200,
#                    node_list,
#                    act_fun = 'relu',
#                    output_fun = 'linear',
#                    opt_fun = 'adam',
#                    cost_fun = 'mean_absolute_error',
#                    dropout_rate = 0.0,
#                    lr_rate = 0.001,
#                    lambd = 0.0001)
#
#encoder.summary()
#history = encoder.fit(train_pot,train_eig, batch_size=64, epochs=50,
#                    validation_data = (val_pot,val_eig)) 
#encoder.save('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoder2.h5')

dec_nodes = [200, 200, 200, 200, 400, 600, 800]
                      
autoencoder = autoencoder_pipeline(1001,
                    200,
                    dec_nodes,
                    chosen_encoder,
                    act_fun = 'relu',
                    output_fun = 'linear',
                    opt_fun = 'adam',
                    cost_fun = 'mean_squared_error',
                    dropout_rate = 0.0,
                    lr_rate = 0.001,
                    lambd = 0.0001)
history = autoencoder.fit(train_pot, train_pot, batch_size=64, epochs=50,
                    validation_data=(val_pot,val_pot))
autoencoder.summary()
#training_plot(history)

#run model selection
#saved_params, saved_rms, saved_models = model_selector(35, 3, 200, 1001)
#file_creator(saved_params, saved_rms, saved_models)
