# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:45:47 2020

@author: Oskari
"""

import sys
sys.path.append('../molecular-state-classes-and-functions/')

#Import custom functions and classes used to manipulate molecular states and Hamiltonian
from functions import *
from classes import *
from quantum_operators import *

import numpy as np
import sympy
from scipy.linalg import expm
from tqdm import tqdm

#%% Define functions

def propagate_state(lens_state_vec, H, QN, B, E, T, N_steps):
    """
    Function that propagates the state of a quantum system defined by 
    Hamiltonian H from t = 0 to t=T. The initial state of the system is the 
    eigenstate of the Hamiltonian H that has the highest overlap with 
    initial_state_vec. The Hamiltonian may include time-dependent EM fields
    defined by E and B.
    
    Note: the function assumes that the E- and B-fields at times t = 0 and 
    t = T are the same. This is done in order to be able to compare the initial
    and final state vectors.
    
    inputs:
    initial_state_vec = defines initial state of system
    H = Hamiltonian of system as function of B and E
    QN = list that defines the basis of the Hamiltonian and state vectors
    B = magnetic field as function of time
    E = electric field as function of time
    T = total time for which to propagate the state
    N_steps = number of timesteps used in the propagation
    
    outputs:
    probability = overlap**2 of the initial and final state vectors
    """
    
    #Find the initial state by diagonalizing the Hamiltonian at t=0 and finding
    #which of the eigenstates is closest to initial_state_vec
    
    #Diagonalize H
    energies0, evecs0 = np.linalg.eigh(H(E(0),B(0)))
    
    #Find the index of the state corresponding to the lens state
    index_ini = find_state_idx(lens_state_vec, evecs0, n=1)
    
    #Find the exact state vector of the initial state
    initial_state_vec = evecs0[:,index_ini[0]:index_ini[0]+1]
    
    #Define initial state
    psi = initial_state_vec
    
    #Calculate timestep
    dt = T/N_steps
    
    #Loop over timesteps to evolve system in time:
    for i in range(1,N_steps+1):
        #Calculate time for this step
        t_i = i*dt 
        
        #Evaluate hamiltonian at this time
        H_i = H(E(t_i), B(t_i))
        
        #Propagate the state vector        
        psi = expm(-1j*2*np.pi*H_i*dt) @ psi
    

    #Determine which eigenstate of the Hamiltonian corresponds to the lens state
    #in the final values of the E- and B-fields
    #Diagonalize H
    energiesT, evecsT = np.linalg.eigh(H(E(T),B(T)))
    
    #Find the index of the state corresponding to the lens state
    index_fin = find_state_idx(lens_state_vec, evecsT, n=1)
    
    #Find the exact state vector that corresponds to the lens state in the given
    #E and B field
    final_state_vec = evecsT[:,index_fin[0]:index_fin[0]+1]
    
    #Calculate overlap between the initial and final state vector
    overlap = np.dot(np.conj(psi).T,final_state_vec)
    
    #Return probability of staying in original state
    return np.abs(overlap)**2
    



def get_E_field(options_dict):
    """
    Function that generates the electric field as a function of time based on
    an options dictionary.
    """

