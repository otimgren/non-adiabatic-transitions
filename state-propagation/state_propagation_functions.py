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
from scipy.linalg.lapack import zheevd

import sys
sys.path.append('./molecular-state-classes-and-functions/')

#Import custom functions and classes used to manipulate molecular states and Hamiltonian
from functions import *
from classes import *
from quantum_operators import *
#%%Define functions

def propagate_state_var_dt(lens_state_vec, H, QN, B, E, T, save_fields=False):
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
    save_fields = Boolean that determines if EM fields as function of time should be saved
    
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
        
    #Loop over timesteps to evolve system in time:
    t = 0
    while t<T:
        #Calculate timestep
        dt = calculate_timestep(t, E)
        
        #Calculate time for this step
        t += dt
        
        #Evaluate hamiltonian at this time
        H_i = H(E(t), B(t))
        
        #Diagonalize the hamiltonian
        D, V = np.linalg.eigh(H_i)
        
        #Propagate the state vector        
        psi = V @ np.diag(np.exp(-1j*D*dt)) @ np.conj(V.T) @ psi
    

    #Determine which eigenstate of the Hamiltonian corresponds to the lens state
    #in the final values of the E- and B-fields
    #Diagonalize H
    energiesT, evecsT = np.linalg.eigh(H(E(t),B(t)))
    
    #Find the index of the state corresponding to the lens state
    index_fin = find_state_idx(lens_state_vec, evecsT, n=1)
    
    #Find the exact state vector that corresponds to the lens state in the given
    #E and B field
    final_state_vec = evecsT[:,index_fin[0]:index_fin[0]+1]
    
    #Calculate overlap between the initial and final state vector
    overlap = np.dot(np.conj(psi).T,final_state_vec)
    
    #Return probability of staying in original state
    return np.abs(overlap)**2

def propagate_state(lens_state_vec, H, QN, B, E, T, N_steps, save_fields=False):
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
    save_fields = Boolean that determines if EM fields as function of time should be saved
    
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
    
    #If fields are to be saved, initialize containers
    if save_fields:
        E_field = np.zeros((3,N_steps))
        B_field = np.zeros((3,N_steps))
        t_array = np.zeros((N_steps))
        
        E_field[:,0] = E(0)
        B_field[:,0] = B(0)

    #Calculate timestep
    dt = T/N_steps
        
    #Loop over timesteps to evolve system in time:
    t = 0
    for n in range(0,N_steps):
        #Calculate time for this step
        t += dt
        
        #Calculate fields
        E_t = E(t)
        B_t = B(t)
        
        #Evaluate hamiltonian at this time
        H_i = H(E_t, B_t)
        
        #Diagonalize the hamiltonian
        D, V, info = zheevd(H_i)
        
        #If zheevd doesn't converge, use numpy instead
        if info != 0:
            D, V = np.linalg.eigh(H_i)
        
        #Propagate the state vector        
        psi = V @ np.diag(np.exp(-1j*D*dt)) @ np.conj(V.T) @ psi
        
        #If fields are saved, add to array
        if save_fields:
            E_field[:,n] = E_t
            B_field[:,n] = B_t
            t_array[n] = t
    

    #Determine which eigenstate of the Hamiltonian corresponds to the lens state
    #in the final values of the E- and B-fields
    #Diagonalize H
    energiesT, evecsT = np.linalg.eigh(H(E(t),B(t)))
    
    #Find the index of the state corresponding to the lens state
    index_fin = find_state_idx(lens_state_vec, evecsT, n=1)
    
    #Find the exact state vector that corresponds to the lens state in the given
    #E and B field
    final_state_vec = evecsT[:,index_fin[0]:index_fin[0]+1]
    
    #Calculate overlap between the initial and final state vector
    overlap = np.dot(np.conj(psi).T,final_state_vec)
    probability = np.abs(overlap)**2
    
    #Initialize dictionary to store results
    results_dict ={"probability":probability}
    
    if save_fields:
        results_dict["E_field"] = E_field
        results_dict["B_field"] = B_field
        results_dict["t_array"] = t_array
    
    #Return probability of staying in original state
    return results_dict
    
def get_Hamiltonian(options_dict): 
    """
    Function that gets the hamiltonian from a file specified in the options
    dictionary and returns it as a function of electric and magnetic fields
    """
    H_fname = options_dict["H_fname"]
    run_dir = options_dict["run_dir"]
    H = make_hamiltonian(run_dir+H_fname)
    
    return H

def E_field_lens(x, z0 = 0, V = 3e4, R = 1.75*0.0254/2, L = 0.60, l = 20e-3):
    """
    A function that gives the electric field due to to the electrostatic lens
    E_vec = electric field vector in V/cm
    x = position (m)
    z0 = center of lens (m)
    V = voltage on electrodes (V)
    R = radius of lens (m)
    L = length of lens (m)
    l = decay length of lens field (m)
    """
    
    #Calculate electric field vector (assumed to be azimuthal and perpendicular to r_vec)
    E_vec = 2*V/R**2 * np.array((-x[1], x[0], 0))
    
    #Scale the field by a tanh function so it falls off outside the lens
    E_vec = E_vec * (np.tanh((x[2]-z0+L/2)/l) - np.tanh((x[2]-z0-L/2)/l))/2
    
    return E_vec/100

#Define a function that gives the Ez component of the lens field as a function of position
def lens_Ez(x, lens_z0, lens_L):
    """
    Function that evaluates the z-component of the electric field produced by the lens based on position
    
    inputs:
    x = position (in meters) where Ex is evaluated (np.array) 
    
    returns:
    E = np.array that only has z-component (in V/cm)
    """
    
    #Determine radial position and the angle phi in cylindrical coordinates
    r = np.sqrt(x[0]**2+x[1]**2)
        
    #Calculate the value of the electric field
    #Calculate radial scaling
    c2 = 13673437.6
    c4 = 9.4893e+09
    radial_scaling = c2*r**2 + c4*r**4
    
    #Angular function
    angular = 2 * x[0]*x[1]/(r**2)
    
    #In z the field varies as a lorentzian
    sigma = 12.811614314258744/1000
    z1 = lens_z0 - lens_L/2
    z2 = lens_z0 + lens_L/2
    z_function = (np.exp(-(x[2]-z1)**2/(2*sigma**2)) - np.exp(-(x[2]-z2)**2/(2*sigma**2)))
    
    E_z = radial_scaling*angular*z_function
    
    return np.array((0,0,E_z))

def E_field_ring(x,z0 = 0, V = 2e4, R = 2.25*0.0254):
    """
    A function that calculates the axial electric field due to a ring electrode
    E_vec = electric field vector in V/m
    x = position (m)
    Q = charge on ring  (V m)
    R = radius of ring (m)
    """
    #Determine z-position
    z = x[2]
    
    #Calculate electric field
    #The electric field is scaled so that for R = 2.25*0.0254m, get a max field
    #of E = 100000 V/m for a voltage of 20 kV
    scaling_factor = (2.25*0.0254)**2/20e3 * (1e5) *3*np.sqrt(3)/2
    mag_E = scaling_factor*(z-z0)/((z-z0)**2 + R**2)**(3/2)*V
    
    #Return the electric field as an array which only has a z-component (approximation)
    return np.array((0,0,mag_E))/100

def E_field_state_prep2(x, E0 = 60, k=100, l = 0.025,z1 = -.1):
    """
    Function that calculates the electric field due to the 2nd state preparation region 
    at position x.

    inputs:
    x = position of molecule in m
    E0 = Value of electric field at center of SP2 (V/cm)
    k = gradient of electric field magnitude (V/cm/m)
    l = rise-length (decay length) of SP2 electric field (m)

    outputs:
    E = vector desecribing electric field due to SP2
    """
    z2 = z1+.2
    E = np.array((0,0,0.5*(E0+k*x[2])*(np.tanh((x[2]-z1)/l)-np.tanh((x[2]-z2)/l))))

    return E




def get_E_field(options_dict):
    """
    Function that generates the electric field due to the lens and rings
    as a function of position based on an options dictionary. 

    returns:
    E_field =   lambda function that returns electric field vector (V/cm) as function of 
                position (m)
    """
    #Get parameters that define the electric field from the options dictionary
    lens_z0 = 0.6/2 + 0.25
    lens_l = options_dict["lens"]["l"]
    lens_L = 0.6
    V1 = options_dict["ring_1"]["V"]
    ring1_z0 = options_dict["ring_1"]["z0"]
    V2 = options_dict["ring_2"]["V"]
    ring2_z0 = options_dict["ring_2"]["z0"]

    #Parameters for ring before SP2
    dz = options_dict["ring_3"]["dz"]
    z0_SP2 = lens_z0+lens_L/2 + dz
    V_SP2 = options_dict["ring_3"]["V"]
    
    #Make a function that gives the electric field as a function of position
    E_field = lambda x: (E_field_ring(x, z0 = ring1_z0, V = V1) 
                         + E_field_ring(x, z0 = ring2_z0, V = V2)
                         + E_field_lens(x, z0 = lens_z0, l = lens_l)
                         + lens_Ez(x, lens_z0, lens_L)
                         + E_field_ring(x, z0 = z0_SP2, V = -V_SP2))
    
    return E_field


def poly_gauss(z, c2, c4, l):     
    """
    Define the functional form to be used for approximating the magnetic fields
    """
    return (1 + c2*z**2 + c4*z**4)*np.exp(-(z/l)**2)

def get_B_field(options_dict):
    """
    Function that generates the magnetic field as a function of position based on
    an options dictionary.
    """
    #Get earth field from options dictionary
    B_earth = np.array(options_dict["B_earth"])
    
    #Check if B-field is supposed to be canceled
    cancel_B_field = options_dict["cancel_B_field"]

    #Get position of B-field zeros from options dictionary
    pos_f = options_dict["B0_pos"]
    
    #Define the start and end z-position of the lens
    z1 = 0.25
    z2 = z1+0.6
    
    l = options_dict["lens"]["l"]
    
    r_B0_1 = np.array((0,0,z1-pos_f*l))
    r_B0_2 = np.array((0,0,z2+pos_f*l))
    
    z1_0 = r_B0_1[2]
    z2_0 = r_B0_2[2]


    #get B-field due to coils as a function of position
    B_poly = lambda r: np.array(((poly_gauss(r[2]-z1_0,-2.93758956, 0.93659248,0.49445160)
                                    +poly_gauss(r[2]-z2_0,-2.93758956, 0.93659248,0.49445160)),
                                (poly_gauss(r[2]-z1_0,-6.95225620, 3.9847e-04,0.25704185)
                                    +poly_gauss(r[2]-z2_0,-6.95225620, 3.9847e-04,0.25704185)),
                                (poly_gauss(r[2]-z1_0,-4.40958415, 20.8698396,0.35204689)
                                    +poly_gauss(r[2]-z2_0,-4.40958415, 20.8698396,0.35204689))))

    if cancel_B_field:                            
        B_field = lambda r: B_earth*(1-B_poly(r)/B_poly(r_B0_1))     

    else:
        B_field = lambda r: B_earth                  
    
    #Return the function B_t
    return B_field
    

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

def calculate_timestep(t, E_t, dt_max = 1e-4, dt_grad = 1e-9, scaling_factor = 1):
    """
    Function for evaluating a suitable timestep to propagate the simulation at a given time
    inputs:
    t = current time
    E_t = electric field as a function of time, used to evaluate gradient of electric field
    scaling_factor = parameter to scale the timesteps if desired
    
    returns:
    dt = timestep for simulation
    """
    
    #Calculate gradient of E
    grad_E = (E_t(t+dt_grad)-E_t(t))/dt_grad
    
    #Calculate dipole moment in hertz
    D_TlF = 4.2282 * 0.393430307 *5.291772e-9/4.135667e-15 # [Hz/(V/cm)]
    
    #Calculate suitable timestep
    dt = 1/np.sqrt(D_TlF * np.linalg.norm(grad_E)) * scaling_factor
    
    #If timestep is larger than the max allowed, set timestep to the max
    if dt > dt_max:
        dt = dt_max
        
    return dt