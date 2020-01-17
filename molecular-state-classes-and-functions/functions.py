# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:39:49 2019

Storing functions needed for the notebook in the same folder in this file.

@author: Oskari
"""

import numpy as np
from classes import UncoupledBasisState, State
import sympy
import pickle


#Function for making a Hamiltonian function from a hamiltonian stored in a file
def make_hamiltonian(path, c1 = 126030.0, c2 = 17890.0,
                     c3 = 700.0, c4 = -13300.0):
    with open(path, 'rb') as f:
        hamiltonians = pickle.load(f)

        #Substitute values into hamiltonian
        variables = [
            sympy.symbols('Brot'),
            *sympy.symbols('c1 c2 c3 c4'),
            sympy.symbols('D_TlF'),
            *sympy.symbols('mu_J mu_Tl mu_F')
        ]
        
        lambdified_hamiltonians = {
            H_name : sympy.lambdify(variables, H_matrix)
            for H_name, H_matrix in hamiltonians.items()
        }
        
        
        
        #Molecular constants
        Brot = 6689920000
        D_TlF = 4.2282 * 0.393430307 *5.291772e-9/4.135667e-15 # [Hz/(V/cm)]
        mu_J = 35 # Hz/G
        mu_Tl = 1240.5 #Hz/G
        mu_F = 2003.63 #Hz/G
        
        H = {
            H_name : H_fn(
                Brot,
                c1, c2, c3, c4,
                D_TlF,
                mu_J, mu_Tl, mu_F
            )
            for H_name, H_fn in lambdified_hamiltonians.items()
            }
        
        Ham = lambda E,B: H["Hff"] + \
            E[0]*H["HSx"]  + E[1]*H["HSy"] + E[2]*H["HSz"] + \
            B[0]*H["HZx"]  + B[1]*H["HZy"] + B[2]*H["HZz"]

        return Ham



def distancematrix(vec1, vec2):
    """simple interpoint distance matrix"""
    v1, v2 = np.meshgrid(vec1, vec2)
    return np.abs(v1 - v2)

#Function for turning a matrix of eigenvectors into a list of state objects
#As input give matrix with columns corresponding to eigenvectors and
#a matrix QN which tells the quantum numbers of each eigenstate
def matrix_to_states(V, QN, E = None):
    #Find dimensions of matrix
    matrix_dimensions = V.shape
    
    #Initialize a list for storing eigenstates
    eigenstates = []
    
    for i in range(0,matrix_dimensions[1]):
        #Find state vector
        state_vector = V[:,i]
        
        data = []
        
        #Get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            data.append((amp, QN[j]))
            
        #Store the state in the list
        state = State(data)
        
        if E is not None:
            state.energy = E[i]
        
        eigenstates.append(state)
        
    
    #Return the list of states
    return eigenstates

def vector_to_state(state_vector, QN, E = None):
    data = []
    
    #Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        data.append((amp, QN[j]))
        
    return State(data)

#Function for tracing the energy of a given state as electromagnetic field
#is varied
#E is vector of eigenenergies, V a matrix with each column the eigenvector
#corresponding to E and state_vec a vector corresponfing to the state to follow
def follow_energy(E,V,state_vec):
    #Initialize array for storing energies at each point
    energy_array = np.zeros(V.shape[0])
    
    for i in range(0,V.shape[0]):     
        #Take dot product between each eigenvector in V and state_vec
        overlap_vector = np.absolute(np.matmul(V[i,:,:].T,state_vec))
        
        #Find which state has the largest overlap:
        index = np.argmax(overlap_vector)
        
        #Store energy
        energy_array[i] = E[i,index]
        
    return energy_array
    
def follow_state(E,V,state_vec):
    #Initialize array for storing energies at each point
    energy_array = np.zeros(V.shape[0])
    state_array = np.zeros((V.shape[0],V.shape[1]))
    
    for i in range(0,V.shape[0]):     
        #Take dot product between each eigenvector in V and state_vec
        overlap_vector = np.absolute(np.matmul(V[i,:,:].T,state_vec))
        
        #Find which state has the largest overlap:
        index = np.argmax(overlap_vector)
        
        #Store energy and state
        energy_array[i] = E[i,index]
        state_array[i,:] = V[i,:,index]
        
    return energy_array, state_array


def follow_state_adiabatic(E,V,state_vec):
    #Initialize array for storing energies at each point
    energy_array = np.zeros(V.shape[0])
    state_array = np.zeros((V.shape[0],V.shape[1]))
    
    V_prev = state_vec
    
    for i in range(0,V.shape[0]):     
        #Take dot product between each eigenvector in V and state_vec
        overlap_vector = np.absolute(np.matmul(np.conj(V[i,:,:].T),V_prev))
        
        #Find which state has the largest overlap:
        index = np.argmax(overlap_vector)
        
        #Store energy and state
        energy_array[i] = E[i,index]
        state_array[i,:] = V[i,:,index]
        
        V_prev = V[i,:,index]
        
    return energy_array, state_array


def follow_states_adiabatic(E,V,state_vecs):
    #Initialize array for storing energies at each point
    energy_array = np.zeros((V.shape[0],V.shape[2]))
    state_array = np.zeros((V.shape[0],V.shape[1],V.shape[2]))
    
    V_prev = state_vecs
    
    for i in range(0,V.shape[0]):
        
        
        V_in = V[i,:,:]
        E_in = E[i,:]
                
        
        #Take dot product between each eigenvector in V and state_vec
        overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T),V_prev))
        
        #Find which state has the largest overlap:
        index = np.argsort(np.argmax(overlap_vectors,axis = 1))
        #Store energy and state
        energy_array[i,:] = E_in[index]
        state_array[i,:,:] = V_in[:,index]
        
        V_prev = V_in[:,index]
        
    return energy_array, state_array


"""
This function determines the index of the state vector most closely corresponding
to an input state vector

state_vecs = array of eigenstates for a Hamiltonian
input_vec = input vector which is compared to the eigenstates
idx = indices of the eigenstates that most closely correspond to the input
n = number of indices to output in order of decreasing overlap
"""

def find_state_idx(input_vec, state_vecs, n = 1):
    overlaps = np.dot(np.conj(input_vec),state_vecs)
    probabilities = overlaps*np.conj(overlaps)
    
    idx = np.argsort(-probabilities)
    
    return idx[0:n]


""" 
Function to reshuffle the eigenvectors and eigenenergies based on a reference
V_in = eigenvector matrix to be reorganized
E_in = energy vector to be reorganized
V_ref = reference eigenvector matrix to be reorganized
V_out = reorganized version of V_in
E_out = reorganized version of E_in
"""
def reorder_evecs(V_in,E_in,V_ref):
    #Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T),V_ref))
    
    #Find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors,axis = 1))
    #Store energy and state
    E_out = E_in[index]
    V_out = V_in[:,index]   
    
    return E_out, V_out


""" 
Function to reshuffle the eigenvectors and eigenenergies based on a reference
when there are degenerate eigenstates.
V_in = eigenvector matrix to be reorganized
E_in = energy vector to be reorganized
V_ref = 3D matrix  of reference eigenvectors that is used to reorder V_in - 
should contain eigenvectors of the last 5 field values to deal with degeneracies 
V_out = reorganized version of V_in
E_out = reorganized version of E_in
"""
def reorder_degenerate_evecs(V_in,E_in,V_ref_n):
    overlap_vectors = []
    #Loop over the refernce evec matrices in V_ref_n
    for V_ref in V_ref_n:
        #Take dot product between each eigenvector in V and state_vec
        overlap_vectors.append(np.absolute(np.matmul(np.conj(V_ref.T),V_in))**2)
        
        #If the ordering of the states is clear (i.e. max element in each
        #overlap vector > some limit), move on to rest of calculation
        if np.all(np.max(overlap_vectors,axis = 0) > 0.9):
            break
        
    
    overlap_vectors = sum(overlap_vectors)
    
    #Find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors,axis = 1))
    #Store energy and state
    E_out = E_in[index]
    V_out = V_in[:,index]   
    
    return E_out, V_out