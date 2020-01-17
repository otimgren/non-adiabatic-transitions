# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:45:47 2020

@author: Oskari
"""

import numpy as np
from scipy.linalg import expm

from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import sympy as sp

from tqdm import tqdm

import sys
sys.path.append('../molecular-state-classes-and-functions/')

#Import custom functions and classes used to manipulate molecular states and Hamiltonian
from functions import *
from classes import *
from quantum_operators import *
#%%Define functions

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
    for i in tqdm(range(1,N_steps+1)):
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
    
def get_Hamiltonian(options_dict): 
    """
    Function that gets the hamiltonian from a file specified in the options
    dictionary and returns it as a function of electric and magnetic fields
    """
    H_fname = options_dict["H_fname"]
    run_dir = options_dict["run_dir"]
    H = make_hamiltonian(run_dir+H_fname)
    
    return H

def get_E_field(options_dict):
    """
    Function that generates the electric field as a function of time based on
    an options dictionary.
    """
    #Get electric field components
    Ex0 = options_dict["E_params"]["Ex0"]
    Ey0 = options_dict["E_params"]["Ey0"]
    Ez0 = options_dict["E_params"]["Ez0"]
    
    #Get the time constant for the lens field from options dictionary
    tau = options_dict["E_params"]["tau"]
    
    #Calculate other time parameters based on tau
    T = 20*tau #s
    t0 = T * 1/4 #s
    t1 = T * 3/4 #s
    
    #Parse the expression given for E as function of time in options dictionary
    t = Symbol('t', real = True)
    scope = locals().copy()
    E_sympy = [parse_expr(E_component, local_dict = {**scope, 'sp':sp}) for E_component in options_dict["E_t"]]
    E_t = lambda t: np.array([lambdify('t',E_component)(t) for E_component in E_sympy])
    
    #Return the function E_t
    return E_t
    
    
def get_B_field(options_dict):
    """
    Function that generates the magnetic field as a function of time based on
    an options dictionary.
    """
    #Get magnetic field components
    Bx0 = options_dict["B_params"]["B0"][0]
    By0 = options_dict["B_params"]["B0"][1]
    Bz0 = options_dict["B_params"]["B0"][2]
    B0 = options_dict["B_params"]["B0"]
    
    #Get the time constant for the lens field from options dictionary
    tau = options_dict["E_params"]["tau"]
    
    #Calculate other time parameters based on tau
    T = 20*tau #s
    tau_B = .05*tau
    deltat = 1.5*tau
    t0 = T * 1/4 #s
    t1 = T * 3/4 #s
    t0_B = t0-2*tau
    t1_B = t1+2*tau
    f = options_dict["B_params"]["f"]
    
    #Parse the expression given for E as function of time in options dictionary
    t = Symbol('t', real = True)
    scope = locals().copy()
    B_sympy = parse_expr(options_dict["B_t"], local_dict = {**scope, 'sp':sp, 'np':np})
    B_t = lambda t: lambdify('t',B_sympy)(t)
    
    #Return the function B_t
    return B_t
    

def generate_QN(Jmin = 0, Jmax = 6, I_Tl = 1/2, I_F = 1/2):
    """
    Function that generates the QN list
    """
    QN = [UncoupledBasisState(J,mJ,I_Tl,m1,I_F,m2)
      for J  in np.arange(Jmin, Jmax+1)
      for mJ in np.arange(-J,J+1)
      for m1 in np.arange(-I_Tl,I_Tl+1)
      for m2 in np.arange(-I_F,I_F+1)
     ]
    
    return QN