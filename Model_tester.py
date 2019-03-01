#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:03:46 2019

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

#data_folder = 'nydata100'
#data_folder = 'nydata200'
#data_folder = 'data1D200'
#data_folder='data1D900'

data_folder = 'nydata200en'

#model_dirr = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/100models/model_0.01846.h5'
#model_dirr = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200models_ny/model_0.01608.h5'
#model_dirr = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200models_encoder1/model_0.01569.h5'
#model_dirr = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/900models_encoder5/model_0.01330.h5'
 
model_dirr = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/autoencoder_fix_nydata200en.h5'

encoder100_ny = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/100encoderny.h5'
encoder200_ny = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoderny.h5'
encoder200_3 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200encoder3.h5'
encoder900_5 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/900encoder5.h5'

polyencoder_200 = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/200polyencoderny.h5'

chosen_encoder = polyencoder_200

def file_loader(folder_name):
    train_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/pot_train_d.txt'
    train_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/eig_train_d.txt'
    val_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/pot_val_d.txt'
    val_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/eig_val_d.txt'
    test_pot_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/pot_test_d.txt'
    test_eig_file = '/home/victor/anaconda3/envs/ML2/scripts/Exjobb/'+ folder_name +'/eig_test_d.txt'
    
    train_pot = np.loadtxt(train_pot_file)
    train_eig = np.loadtxt(train_eig_file)
    val_pot = np.loadtxt(val_pot_file)
    val_eig = np.loadtxt(val_eig_file)
    test_pot = np.loadtxt(test_pot_file)
    test_eig = np.loadtxt(test_eig_file)
    
    return train_pot, train_eig, val_pot, val_eig, test_pot, test_eig

train_pot, train_eig, val_pot, val_eig, test_pot, test_eig = file_loader(data_folder)
model= load_model(model_dirr)
encoder = load_model(chosen_encoder)


decoder = model.layers[-1]

method = getattr(keras.optimizers, 'adam')
decoder.compile(optimizer = method(lr = 1E-5),
                  loss = 'mean_squared_error',
                 metrics=['accuracy', 'mse', 'mae'])

model_encoder = Model(model.inputs, model.layers[-2].output)
model_encoder.compile(optimizer = method(lr = 1E-5),
                  loss = 'mean_squared_error',
                 metrics=['accuracy', 'mse', 'mae'])


#decoder.summary()

encoder_test = encoder.predict(test_pot)
#model_enc_test = model_encoder.predict(test_pot)
#decoder_test = decoder.predict(test_eig)
#pred_test = model.predict(test_pot)

for i in range(10):
    plt.figure()
#    plt.plot(range(len(pred_test[i])),pred_test[i],  color = 'r', label = 'Autoencoder prediction')
    plt.plot(range(len(test_pot[i])),test_pot[i],  color = 'b', label = 'Actual potential')
#    plt.plot(range(len(decoder_test[i])),decoder_test[i],  color = 'g', label = 'Decoder potential')
#for i in range(10):
    plt.figure()
    plt.plot(range(len(encoder_test[i])),list(reversed(encoder_test[i])), marker = '.', color = 'r', label= 'Encoder predictions')
    plt.plot(range(len(test_eig[i])),list(reversed(test_eig[i])), marker = '.', color = 'b', label= 'Actual eigenvalues')
#    plt.plot(range(len(model_enc_test[i])),list(reversed(model_enc_test[i])), marker = '.', color = 'g', label= 'Extracted encoder predictions')


#plot images for text

#plt.figure()   
#plt.plot(range(len(test_pot[7])),test_pot[7], color = 'b', label = 'Actual potential')
#plt.plot(range(len(fitted_test[7])),fitted_test[7],  color = 'r', label = 'Autoencoder prediction')
#plt.plot(range(len(decoder_test[7])),decoder_test[7],  color = 'g', label = 'Decoder prediction')
#plt.title('A potential from data-set scaled-200set-1')
#plt.xlabel('Grid point')
#plt.ylabel('Potential value [eV]')
#plt.legend()
#plt.savefig('/home/victor/Documents/exjobb/bilder/autoenc_potential_polybad.eps', format='eps', dpi=1000)

#plt.figure()   
#plt.figure()
#plt.plot(range(len(encoder_test[7])),list(reversed(encoder_test[7])), marker = '.', color = 'r', label= 'Encoder predictions')
#plt.plot(range(len(test_eig[7])),list(reversed(test_eig[7])), marker = '.', color = 'b', label= 'Actual eigenvalues')
#plt.plot(range(len(model_enc_test[7])),list(reversed(model_enc_test[7])), marker = '.', color = 'g', label= 'Extracted encoder predictions')
#plt.title('Predicted and actual eigenvalues')
#plt.xlabel('Eigenvalue number')
#plt.ylabel('Eigenvalue value [eV]')
#plt.legend()
#plt.savefig('/home/victor/Documents/exjobb/bilder/autoenc_eigs_polybad.eps', format='eps', dpi=1000)
