# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:22:44 2019

@author: Oskari

This file contains functions used in the data analysis for characterizing
the molecular beam source in CeNTREX.

"""

#Import packages
import numpy as np
from tqdm import tqdm_notebook
import glob
from sympy.physics.wigner import wigner_3j, wigner_6j, clebsch_gordan
import sys
sys.path.append('../molecular-state-classes-and-functions/')
from classes import CoupledBasisState
from sympy import S
from datetime import datetime, timedelta


#### Functions
#%%
##### Calculating molecular beam intensity ####
#Function for evaluating scattering cross section given
def calculate_cross_section(ground_state, excited_state, B_state_eigenstates, gamma, detuning, wavelength):
    
    #Find branching ratio needed in calculating the cross section
    branching_ratio = calculate_branching_ratio(ground_state, excited_state, B_state_eigenstates)
    
    #Calculate the cross section
    F = float(ground_state.F)
    Fprime = float(excited_state.F)
    cross_section = wavelength**2/(2*np.pi)*(2*Fprime+1)/(2*F+1)*branching_ratio*(gamma/2)**2/((gamma/2)**2 + detuning**2)
    
    return float(cross_section)


#Defining a utility function that can be used to turns floats into rational numbers in sympy
def rat(number):
    return S(str(number),rational = True)


#Function for evaluation the electric dipole matrix element between a ground state and excited state
def calculate_microwave_ED_matrix_element(ground_state, excited_state,reduced = True, pol_vec = np.array((0,0,1))):
    #Find quantum numbers for ground state
    J = float(ground_state.J)
    F1 = float(ground_state.F1)
    F = float(ground_state.F)
    mF = float(ground_state.mF)
    I1 = float(ground_state.I1)
    I2 = float(ground_state.I2)
    
    #Find quantum numbers of excited state
    Jprime = float(excited_state.J)
    F1prime = float(excited_state.F1)
    Fprime = float(excited_state.F)
    mFprime = float(excited_state.mF)
    
    #Calculate reduced matrix element
    M_r = (np.sqrt(float((2*F1+1) * (2*F1prime+1) * (2*F+1)* (2*Fprime+1))) * float(wigner_6j(Jprime, F1prime,1/2,F1,J,1))
               * float(wigner_6j(F1prime, Fprime,1/2,F,F1,1)) * np.sqrt(float((2*J+1) * (2*Jprime+1))) 
               *(-1)**(F1prime+J+Fprime+F1+1)
               * float(wigner_3j(J,1,Jprime,0,0,0) * (-1)**J))
    
    if reduced:
        return float(M_r)
    else:
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        prefactor = 0
        for p in range(-1,2):
            prefactor +=  (-1)**(p+F-mF) * p_vec[p] *  float(wigner_3j(F,1,Fprime,-mF,-p,mFprime))
        
        
        return prefactor*float(M_r)
    
    
#Function for evaluation the electric dipole matrix element between a ground state and excited state in uncoupled basis
def calculate_microwave_ED_matrix_element_uncoupled(ground_state, excited_state,reduced = True, pol_vec = np.array((0,0,1))):
    #Find quantum numbers for ground state
    J = float(ground_state.J)
    mJ = float(ground_state.mJ)
    I1 = float(ground_state.I1)
    m1 = float(ground_state.m1)
    I2 = float(ground_state.I2)
    m2 = float(ground_state.m2)
    
    #Find quantum numbers of excited state
    Jprime = float(excited_state.J)
    mJprime = float(excited_state.mJ)
    I1prime = float(excited_state.I1)
    m1prime = float(excited_state.m1)
    I2prime = float(excited_state.I2)
    m2prime = float(excited_state.m2)
    
    #Calculate reduced matrix element
    M_r = (wigner_3j(J,1,Jprime,0,0,0) * np.sqrt((2*J+1)*(2*Jprime+1)) 
            * float(I1 == I1prime and m1 == m1prime 
                    and I2 == I2prime and m2 == m2prime))
    
    
    
    if reduced:
        return float(M_r)
    else:
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        prefactor = 0
        for p in range(-1,2):
            prefactor +=  (-1)**(p-mJ) * p_vec[p] *  float(wigner_3j(J,1,Jprime,-mJ,-p,mJprime))
        
        
        return prefactor*float(M_r)

#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_superposition(ground_state, excited_state, X_state_eigenstates):
    #Find quantum numbers for ground state
    J = ground_state.J
    F1 = ground_state.F1
    F = ground_state.F
    I1 = ground_state.I1
    I2 = ground_state.I2
    
    #Find quantum numbers of excited state and determine what the 'real' mixed eigenstate is by looking it up from a 
    #dictionary
    Jprime = excited_state.J
    F1prime = excited_state.F1
    Fprime = excited_state.F
    Pprime = excited_state.P
    
    
    #Generate the name of the state
    ground_state_name = "|J = %s, F1 = %s, F = %s, mF = 0, I1 = %s, I2 = %s>"%(rat(J),rat(F1),rat(F)
                                                                                ,rat(I1),rat(I2))
    excited_state_name = "|J = %s, F1 = %s, F = %s, mF = 0, I1 = %s, I2 = %s>"%(rat(Jprime),rat(F1prime),rat(Fprime)
                                                                                ,rat(I1),rat(I2))
    
    #Find states in dictionary
    ground_state_mixed = X_state_eigenstates[ground_state_name]
    excited_state_mixed = X_state_eigenstates[excited_state_name]
        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M_r = 0
    
    for amp1, basis_state1 in ground_state_mixed.data:
        for amp2, basis_state2 in excited_state_mixed.data:
            M_r += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element(basis_state1, basis_state2)
        
    return M_r


#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_mixed_state(ground_state, excited_state,reduced = True,pol_vec = np.array((0,0,1))):        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M = 0
    
    for amp1, basis_state1 in ground_state.data:
        for amp2, basis_state2 in excited_state.data:
            M += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element(basis_state1, basis_state2,reduced,pol_vec)
        
    return M


#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_mixed_state_uncoupled(ground_state, excited_state,reduced = True,pol_vec = np.array((0,0,1))):        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M = 0
    
    for amp1, basis_state1 in ground_state.data:
        for amp2, basis_state2 in excited_state.data:
            M += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element_uncoupled(basis_state1, basis_state2,reduced,pol_vec)
        
    return M
    