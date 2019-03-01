#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 

@author: victor
"""

import numpy as np
from scipy.sparse import linalg as ln
from scipy import constants as const
import scipy as sp
import random
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.sparse as sparse

grid_dim = 1E-9
grid_points = 1000 #Ger en gridpoint mer än vad som anges
dx = grid_dim/(grid_points+1)
grid = np.array([1*i/grid_points for i in range(grid_points+1)])
random.seed(4)
matrix_const = (const.hbar**2)/(2*const.m_e*grid_dim**2)

#matrix_const = 1



def solver1D(f_grid, nbr_of_eigs=grid_points-1, boundary_condition='d'):
    f_grid = f_grid*const.e
    pot_matrix = sparse.diags(f_grid, 0, (len(f_grid),len(f_grid)))
    
    if(boundary_condition=='p'):
        #periodiska villkor
        diff_matrix = sparse.diags([1, 1, -2, 1, 1], 
                                   [-(len(f_grid)-1), -1, 0, 1, len(f_grid)-1], 
                                   (len(f_grid), len(f_grid)))    
    elif(boundary_condition=='n'):
        #neumann villkor, derivatan = 0
        sup = np.ones(len(f_grid-1))
        sup[0] = 2
        sub = np.ones(len(f_grid-1))
        sub[-1] = 2
        diag = -2 * np.ones(len(f_grid))
        diff_matrix = sparse.diags([sub, diag, sup], [-1, 0, 1], (len(f_grid), len(f_grid)))
    elif(boundary_condition=='d'):
        #dirichlet villkor, värdet = 0
        diff_matrix = sparse.diags([1, -2, 1], [-1, 0, 1], (len(f_grid), len(f_grid)))
        diff_matrix.toarray()         
    
    lh_matrix = (diff_matrix * matrix_const) - (pot_matrix)   
    eig_val = -ln.eigs(lh_matrix, nbr_of_eigs, which='LR', return_eigenvectors=False)
    #print(lh_matrix.toarray())
    eig_val = eig_val / const.e
    return eig_val
    
def grid_generator(nbr_of_func, nbr_of_eigs, periodic_bc):
    grid_matrix = np.zeros((nbr_of_func, len(grid)))
    eig_matrix = np.zeros((nbr_of_func, nbr_of_eigs))
    for i in range(nbr_of_func):
        #potential_type = random.randint(1,3)
        potential_type = 1
        if (potential_type==1):
            grid_matrix[i] = polynomial_generator()
        else:
            grid_matrix[i] = well_perturbation_generator()
        eig_vec = solver1D(grid_matrix[i], nbr_of_eigs, 'p')
        eig_matrix[i] = eig_vec
    return grid_matrix, eig_matrix   
    
def polynomial_generator():
    pot_vec = np.zeros(len(grid))
    r1 = random.randint(1,4)*2
    #r3 = random.uniform(2,3)
    r3 = 2 * random.uniform(5,6)**(1/r1)
    r_val = np.zeros((r1, 3))
    r_val[0] = np.array([r1, 0.5, r3])
    for i in range(1,r1):
        r2 = random.gauss(0.5,0.1)
        r3 = random.uniform(-1,1)
        r_val[i] = np.array([i, r2, r3])
    for j in range(len(pot_vec)):
        for k in range(r1):
            pot_vec[j] = pot_vec[j] + (r_val[k,2]*(grid[j]-r_val[k,1]))**r_val[k,0]
    return pot_vec

def well_perturbation_generator():
    pot_vec = np.zeros(len(grid))
    wall_height = 6
    pot_vec[0] = wall_height
    pot_vec[-1] = wall_height
    perturbation_type = random.randint(1,2)
    if(perturbation_type == 1):
        r_freq = random.randint(1,10)
        r_amp = random.uniform(-np.floor(wall_height/4),np.floor(wall_height/4))
        p_func = lambda x: r_amp*np.sin(2*const.pi*x*r_freq) + np.abs(r_amp)
        pot_vec = pot_vec + p_func(grid) 
    elif(perturbation_type == 2):
        max_points = np.floor(grid_points/10)
        p_points = random.randint(1,max_points)
        p_height = random.uniform(1,wall_height/2)
        p_start = random.randint(2,len(grid)-(p_points+2))
        for i in range(p_points):
            pot_vec[p_start+i] = p_height
    return pot_vec


def file_creator(grid_matrix, eig_matrix, file_name_pot, file_name_eig):
    pot_writer = open('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/' + file_name_pot, 'w')
    eig_writer = open('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/' + file_name_eig, 'w') 
    for i in range(len(grid_matrix)):
        for j in range(len(grid_matrix[i])):
            pot_writer.write(str(grid_matrix[i,j]) + ' ')
        for k in range(len(eig_matrix[i])):
            eig_writer.write(str(eig_matrix[i,k]) + ' ')
        pot_writer.write('\n')
        eig_writer.write('\n')
    print('file done')
        
def generation(eigenvalues, boundary):
    grid_matrix, eig_matrix = grid_generator(10000, eigenvalues, boundary)
    file_creator(grid_matrix[8000:9000], eig_matrix[8000:9000], 'pot_val_'+ boundary + '.txt', 'eig_val_'+ boundary + '.txt')
    file_creator(grid_matrix[:8000], eig_matrix[:8000], 'pot_train_'+ boundary + '.txt', 'eig_train_' + boundary + '.txt')
    file_creator(grid_matrix[9000:], eig_matrix[9000:], 'pot_test_'+ boundary + '.txt', 'eig_test_'+ boundary + '.txt')
    
#test_func_grid, test_eig_grid = grid_generator(10,50,'d')
#plt.figure()
#plt.plot(list(reversed(range(len(test_eig_grid[1])))),test_eig_grid[1], '.')
#plt.figure()
#plt.plot(range(len(test_func_grid[1])),test_func_grid[1], 'r')

#plt.plot(list(reversed(range(len(test_eig_grid[0])))), test_eig_grid[1], 'y')
#plt.plot(list(reversed(range(len(test_eig_grid[0])))), test_eig_grid[2], 'b')
#plt.plot(list(reversed(range(len(test_eig_grid[0])))), test_eig_grid[3], 'g')
#plt.plot(list(reversed(range(len(test_eig_grid[0])))), test_eig_grid[4], 'k')

#for pot_vec in test_func_grid:
#    plt.plot(grid,pot_vec)
#plt.title('10 samples of radomly generated potentials')
#plt.xlabel('Spatial coordinate (1000 grid points)')
#plt.ylabel('Potential value')
#plt.savefig('/home/victor/Documents/exjobb/bilder/potential_image.eps', format='eps', dpi=1000)
#
#test_func = lambda x: 24*(x-0.5)**2
#test_grid = test_func(grid)
##test_grid = np.zeros(len(grid))
#
##test_grid = (test_grid-6) * const.e
#test_eigs = solver1D(test_grid, 20, 'p')
#plt.figure()
#plt.plot(range(len(test_grid)),test_grid, 'k')
#plt.figure()
#plt.plot(list(reversed(range(len(test_eigs)))),test_eigs, '.')
#plt.title('First 100 eigenvalues of harmonic potential(Dirichlet conditions)')
#plt.xlabel('Eigenvalue number')
#plt.ylabel('Eigenvalue value')
##plt.savefig('/home/victor/Documents/exjobb/bilder/eigs_d_image.eps', format='eps', dpi=1000)
#
#plt.figure()
#plt.plot(grid,test_grid, 'r')
#plt.title('Harmonic potential used')
#plt.xlabel('Space coordinate')
#plt.ylabel('Potential value')
#plt.savefig('/home/victor/Documents/exjobb/bilder/harmonic.eps', format='eps', dpi=1000)

#as_test = np.zeros(len(grid))
#as_test[30:35] = 3
#as_test[16:21] = 3
#as_eigs = solver1D(grid, grid_points-1, 'd')
##plt.figure()
##plt.plot(range(len(as_eigs)),as_eigs, '.')
#plt.figure()
#plt.plot(grid,as_test, 'r')

#putting autoencoder output into solver and comparing
decoder_output = np.loadtxt('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/poly200autoencodertest.txt')
testing_eigs = np.loadtxt('/home/victor/anaconda3/envs/ML2/scripts/Exjobb/nydata200en/eig_test_d.txt')
for i in range(10):
    decoder_eigs = solver1D(decoder_output[i], nbr_of_eigs=200)
    plt.figure()
    plt.plot(range(len(decoder_eigs)), list(reversed(decoder_eigs)), marker = '.', color = 'r')
    plt.plot(range(len(testing_eigs[i])), list(reversed(testing_eigs[i])), marker = '.', color = 'b')

#generation(200, 'd')
