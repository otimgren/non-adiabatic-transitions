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
from sympy.physics.wigner import wigner_3j, wigner_6j, clebsch_gordan
from sympy import N



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
        
        # Values for rotational constant are from "Microwave Spectral tables: Diatomic molecules" by Lovas & Tiemann (1974). 
        # Note that Brot differs from the one given by Ramsey by about 30 MHz.
        B_e = 6.689873e9
        alpha = 45.0843e6
        Brot = B_e - alpha/2
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
        
        Ham = lambda E,B: 2*np.pi*(H["Hff"] + \
            E[0]*H["HSx"]  + E[1]*H["HSy"] + E[2]*H["HSz"] + \
            B[0]*H["HZx"]  + B[1]*H["HZy"] + B[2]*H["HZz"])

        return Ham


def calculate_microwave_ME(state1, state2, reduced = False, pol_vec = np.array((0,0,1))):
    """
    Function that evaluates the microwave matrix element between two states, state1 and state2, for a given polarization
    of the microwaves
    
    inputs:
    state1 = an UncoupledBasisState object
    state2 = an UncoupledBasisState object
    reduced = boolean that determines if the function returns reduced or full matrix element
    pol_vec = np.array describing the orientation of the microwave polarization in cartesian coordinates
    
    returns:
    Microwave matrix element between state 1 and state2
    """
    
    #Find quantum numbers for ground state
    J = float(state1.J)
    mJ = float(state1.mJ)
    I1 = float(state1.I1)
    m1 = float(state1.m1)
    I2 = float(state1.I2)
    m2 = float(state1.m2)
    
    #Find quantum numbers of excited state
    Jprime = float(state2.J)
    mJprime = float(state2.mJ)
    I1prime = float(state2.I1)
    m1prime = float(state2.m1)
    I2prime = float(state2.I2)
    m2prime = float(state2.m2)
    
    #Calculate reduced matrix element
    M_r = (N(wigner_3j(J,1,Jprime,0,0,0)) * np.sqrt((2*J+1)*(2*Jprime+1)) 
            * float(I1 == I1prime and m1 == m1prime 
                    and I2 == I2prime and m2 == m2prime))
    
    #If desired, return just the reduced matrix element
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
    
    
def make_H_mu(J1, J2, omega_mu, QN, pol_vec = np.array((0,0,1))):
    """
    Function that generates Hamiltonian for microwave transitions between J1 and J2 (all hyperfine states) for given
    polarization of microwaves. Rotating wave approximation is applied implicitly by only taking the exp(+i*omega*t) part
    of the cos(omgega*t) into account
    
    inputs:
    J1 = J of the lower rotational state being coupled
    J2 = J of upper rotational state being coupled
    QN = Quantum numbers of each index (defines basis fo matrices and vectors)
    pol_vec = vector describing polarization 
    
    returns:
    H_mu = Hamiltonian describing coupling between 
    
    """
    #Figure out how many states there are in the system
    N_states = len(QN) 
    
    #Initialize a Hamiltonian
    H_mu = np.zeros((N_states,N_states), dtype = complex)
    
    
    #Start looping over states and calculate microwave matrix elements between them
    for i in range(0, N_states):
        state1 = QN[i]
        
        for j in range(i, N_states):
            state2 = QN[j]
            
            #Check that the states have the correct values of J
            if (state1.J == J1 and state2.J == J2) or (state1.J == J2 and state2.J == J1):
                #Calculate matrix element between the two states
                H_mu[i,j] = (calculate_microwave_ME(state1, state2, reduced=False, pol_vec=pol_vec))
                
    #Make H_mu hermitian
    H_mu = (H_mu + np.conj(H_mu.T)) - np.diag(np.diag(H_mu))
    
    #Convert H_mu into a lambda function
    H_mu_fun = lambda t: H_mu * np.exp(+1j*omega_mu*t)
    
    #return the hamiltonian matrix as a function of t
    return H_mu_fun

def make_transform_matrix(J1, J2, omega_mu, QN, I1 = 1/2, I2 = 1/2):
    """
    Function that generates the transformation matrix that transforms the Hamiltonian to the rotating frame
    
    inputs:
    J1 = J of lower energy rotational state
    J2 = J of higher energy rotational state
    
    returns:
    U = unitary transformation matrix
    """
    
    #Starting and ending indices of the part of the matrix that has exp(i*omega*t)
    J2_start = int((2*I1+1)*(2*I2+1)*(J2)**2)
    J2_end = int((2*I1+1)*(2*I2+1)*(J2+1)**2)
        
    #Generate the transformation matrices
    D = np.diag(np.concatenate((np.zeros((J2_start)), 
                          -omega_mu * np.ones((J2_end - J2_start)),
                          np.zeros((len(QN)-J2_end)))))
    
    U = lambda t: np.diag(np.concatenate((np.ones((J2_start)), 
                          np.exp(-1j*(omega_mu)*t) * np.ones((J2_end - J2_start)), 
                          np.ones(len(QN)-J2_end))))
    
    
    return U, D


def transform_to_rotating_frame(H, U, D):
    """
    Function that transforms a Hamiltonian H to a rotating basis defined by the 
    unitary transformation matrix U
    """
    
    #Determine the effective hamiltonian in the rotating frame
    Heff = lambda t: np.conj(U(t).T) @ H(t) @ U(t) + D
    
    return Heff


def distancematrix(vec1, vec2):
    """simple interpoint distance matrix"""
    v1, v2 = np.meshgrid(vec1, vec2)
    return np.abs(v1 - v2)

def make_QN(Jmin, Jmax, I1, I2):
    """
    Function that generates a list of quantum numbersfor TlF
    """
    QN = [UncoupledBasisState(J,mJ,I1,m1,I2,m2)
      for J  in np.arange(Jmin, Jmax+1)
      for mJ in np.arange(-J,J+1)
      for m1 in np.arange(-I1,I1+1)
      for m2 in np.arange(-I2,I2+1)
     ]
    
    return QN

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
    
    if n>1:
        idx = np.argsort(-probabilities)
        
        return idx[0:n]
    
    if n == 1:
        idx = np.argmax(probabilities)
        
        return idx



def find_state_idx_from_state(H, reference_state, QN):
    """
    This function determines the index of the state vector most closely corresponding
    to an input state 
    
    H = Hamiltonian whose eigenstates the input state is compared to
    refernce_state = state whose index needs to be determined
    idx = index of the eigenstate that most closely corresponds to the input
    """
    
    #Determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)
    
    #Find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)    
    
    
    
    overlaps = np.dot(np.conj(reference_state_vec),V)
    probabilities = overlaps*np.conj(overlaps)
    
    idx = np.argmax(probabilities)
    
    return idx

def find_closest_state(H, reference_state, QN):
    """
    Function that finds the eigenstate of the Hamiltonian H that most closely corresponds to reference state.
    
    inputs:
    H = hamiltonian whose eigenstates reference_state is compared to
    reference_state = state which we want find in eigenstates of H
    
    returns:
    state = eigenstate of H closest to reference state
    """
    
    #Make a state vector for reference state
    reference_state_vec = reference_state.state_vector(QN)
    
    #Find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)
    
    #Find out which of the eigenstates of H corresponds to reference state
    state_index = find_state_idx(reference_state_vec,V,n=1)
    
    #Find state vector of state corresponding to reference
    state_vec = V[:,state_index:state_index+1]
    
    #return the state
    state = matrix_to_states(state_vec,QN)[0]
    
    return state


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