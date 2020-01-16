# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:23:42 2019

Defining classes that are used in the notebook located in the same folder.
Mostly these are classes that define quantum states in different bases.
Make sure to get the latest version from GitHub 
otimgren/molecular-state-classes-and-function

@author: Oskari
"""


import numpy as np
import sympy as sp
from sympy.physics.quantum.cg import CG
from sympy import S
from sympy import N

#Defining a utility function that can be used to trun floats into rational numbers in sympy
def rat(number):
    return S(str(number),rational = True)

class CoupledBasisState:
    # constructor
    def __init__(self, F, mF, F1, J, I1, I2, P = 1, electronic_state = 0, energy = 0):
        self.F, self.mF  = F, mF
        self.F1 = F1
        self.J = J
        self.I1 = I1
        self.I2 = I2
        self.isCoupled = True
        self.isUncoupled = False
        self.P = P
        self.electronic_state = electronic_state
        self.energy = energy
        
    
    # equality testing
    def __eq__(self, other):
        return self.F==other.F and self.mF==other.mF \
                     and self.I1==other.I1 and self.I2==other.I2 \
                     and self.F1==other.F1 and self.J==other.J

    # inner product
    def __matmul__(self, other):
        if other.isCoupled:
            if self == other:
                return 1
            else:
                return 0
        else:
           return State([(1,other)])@self.transform_to_uncoupled()

    # superposition: addition
    def __add__(self, other):
        if self == other:
            return State([ (2,self) ])
        else:
            return State([
                (1,self), (1,other)
            ])

    # superposition: subtraction
    def __sub__(self, other):
        return self + (-1)*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State([ (a, self) ])

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a
    
    
    # methods
    #Convenience function to print out the quantum numbers of the basis state
    def print_quantum_numbers(self, printing =True):
        F  = S(str(self.F),rational = True)
        mF  = S(str(self.mF),rational = True)
        F1 = S(str(self.F1),rational = True)
        J = S(str(self.J),rational = True)
        I1 = S(str(self.I1),rational = True)
        I2 = S(str(self.I2),rational = True)
        if printing == True:
            print("|J = %s, F1 = %s, F = %s, mF = %s, I1 = %s, I2 = %s>"%(J,F1,F,mF,I1,I2))
        return "|J = %s, F1 = %s, F = %s, mF = %s, I1 = %s, I2 = %s>"%(J,F1,F,mF,I1,I2)
    
    #A method to transform from coupled to uncoupled basis
    def transform_to_uncoupled(self):
        F = self.F 
        mF  = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        
        mF1s = np.arange(-F1,F1+1,1)
        mJs = np.arange(-J,J+1,1)
        m1s = np.arange(-I1,I1+1,1)
        m2s = np.arange(-I2,I2+1,1)
    
        uncoupled_state = State() 
        
        for mF1 in mF1s:
            for mJ in mJs:
                for m1 in m1s:
                    for m2 in m2s:
                        amp = CG(J, mJ, I1, m1, F1, mF1).doit()*CG(F1, mF1, I2, m2, F, mF).doit()
                        basis_state = UncoupledBasisState(J, mJ, I1, m1, I2, m2)
                        uncoupled_state = uncoupled_state + State([(amp, basis_state)])
        
        return uncoupled_state.normalize()
    
    #Makes the basis state into a state
    def make_state(self):
        return State([(1,self)])
    
    #Find energy of state given a list of energies and eigenvecotrs and basis QN
    def find_energy(self,energies,V,QN):
        
        #Convert state to uncoupled basis
        state = self.transform_to_uncoupled()
        
        #Convert to a vector that can be multiplied by the evecs to determine overlap 
        state_vec = np.zeros((1,len(QN)))
        for i, basis_state in enumerate(QN):
            amp = State([(1,basis_state)])@state
            state_vec[0,i] = amp
        
        coeffs = np.multiply(np.dot(state_vec,V),np.conjugate(np.dot(state_vec,V)))
        energy = np.dot(coeffs, energies)
        
        
        self.energy = energy
        return energy

#Class for uncoupled basis states
class UncoupledBasisState:
    # constructor
    def __init__(self, J, mJ, I1, m1, I2, m2):
        self.J, self.mJ  = J, mJ
        self.I1, self.m1 = I1, m1
        self.I2, self.m2 = I2, m2
        self.isCoupled = False
        self.isUncoupled = True

    # equality testing
    def __eq__(self, other):
        return self.J == other.J and self.mJ==other.mJ \
                            and self.I1==other.I1 and self.I2==other.I2 \
                            and self.m1==other.m1 and self.m2==other.m2


    # inner product
    def __matmul__(self, other):
        if other.isUncoupled:
            if self == other:
                return 1
            else:
                return 0
        else:
           return State([(1,self)])@other.transform_to_uncoupled()

    # superposition: addition
    def __add__(self, other):
        if self == other:
            return State([ (2,self) ])
        else:
            return State([
                (1,self), (1,other)
            ])

    # superposition: subtraction
    def __sub__(self, other):
        return self + (-1)*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State([ (a, self) ])

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a
    
    def print_quantum_numbers(self):
        J, mJ  = S(str(self.J),rational = True), S(str(self.mJ),rational = True)
        I1 = S(str(self.I1),rational = True)
        m1 = S(str(self.m1),rational = True)
        I2 = S(str(self.I2),rational = True)
        m2 = S(str(self.m2),rational = True)
        print("|J = %s, mJ = %s, I1 = %s, m1 = %s, I2 = %s, m2 = %s>"%(J,mJ,I1,m1,I2,m2))   
        
    #Makes the basis state into a state
    def make_state(self):
        return State([(1,self)])
    
    #Method for converting to coupled basis
    def transform_to_coupled(self):
        #Determine quantum numbers
        J = self.J
        mJ = self.mJ
        I1 = self.I1
        m1 = self.m1
        I2 = self.I1
        m2 = self.m2
        
        #Determine what mF has to be
        mF = mJ + m1 + m2
        
        uncoupled_state = self
        
        data = []
        
        #Loop over possible values of F1, F and m_F
        for F1 in np.arange(J-I1, J+I1+1):
            for F in np.arange(F1-I2, F1+I2+1):
                if np.abs(mF) <= F:
                    coupled_state = CoupledBasisState(F,mF,F1,J,I1,I2)
                    amp = uncoupled_state @ coupled_state
                    data.append((amp,coupled_state))
                    
        return State(data)
    

        

#Define a class for superposition states        
class State:
    # constructor
    def __init__(self, data=[], remove_zero_amp_cpts=True, energy = 0, name = None):
        # check for duplicates
        for i in range(len(data)):
            amp1,cpt1 = data[i][0], data[i][1]
            for amp2,cpt2 in data[i+1:]:
                if cpt1 == cpt2:
                    raise AssertionError("duplicate components!")
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp,cpt) for amp,cpt in data if amp!=0]
        else:
            self.data = data
        # for iteration over the State
        self.index = len(self.data)
        #Store energy of state
        self.energy = energy
        #Give the state a name if desired
        self.name = name
        

    # superposition: addition
    # (highly inefficient and ugly but should work)
    def __add__(self, other):
        data = []
        # add components that are in self but not in other
        for amp1,cpt1 in self.data:
            only_in_self = True
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
            if only_in_self:
                data.append((amp1,cpt1))
        # add components that are in other but not in self
        for amp1,cpt1 in other.data:
            only_in_other = True
            for amp2,cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
            if only_in_other:
                data.append((amp1,cpt1))
        # add components that are both in self and in other
        for amp1,cpt1 in self.data:
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1+amp2,cpt1))
        return State(data)
                
    # superposition: subtraction
    def __sub__(self, other):
        return self + -1*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State( [(a*amp,psi) for amp,psi in self.data] )

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a
    
    # scalar division (psi / a)
    def __truediv__(self, a):
        return self * (1/a)
    
    # negation
    def __neg__(self):
        return -1 * self
    
    # inner product
    def __matmul__(self, other):
        result = 0
        for amp1,psi1 in self.data:
            for amp2,psi2 in other.data:
                result += amp1.conjugate()*amp2 * (psi1@psi2)
        return result

    # iterator methods
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]
    
    # direct access to a component
    def __getitem__(self, i):
        return self.data[i]
    
    #Some utility functions
    #Function for normalizing states
    def normalize(self):
        data = []
        N = sp.sqrt(self@self)
        for amp, basis_state in self.data:
            data.append([amp/N, basis_state])
            
        return State(data)
    
    
    #Function that displays the state as a sum of the basis states
    def print_state(self, tol = 0.1):
         for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                if np.real(complex(amp))>0: print('+', end ="")
                if basis_state.isCoupled:
                    F = rat(basis_state.F)
                    mF = rat(basis_state.mF)
                    F1 = rat(basis_state.F1)
                    J = rat(basis_state.J)
                    I1 = rat(basis_state.I1)
                    I2 = rat(basis_state.I2)
                    print('{:.4f}'.format(complex(amp))+' x '+"|J = %s, F1 = %s, F = %s, mF = %s, I1 = %s, I2 = %s>"\
                          %(J,F1,F,mF,I1,I2))
                elif basis_state.isUncoupled:
                    J = rat(basis_state.J)
                    mJ = rat(basis_state.mJ)
                    I1 = rat(basis_state.I1)
                    m1 = rat(basis_state.m1)
                    I2 = rat(basis_state.I2)
                    m2 = rat(basis_state.m2)
                    print('{:.4f}'.format(complex(amp))+' x '+"|J = %s, mJ = %s, I1 = %s, m1 = %s, I2 = %s, m2 = %s>"\
                          %(J,mJ,I1,m1,I2,m2))
                 
   
    #Function that returns state vector in uncoupled basis  
    def state_vector(self,QN):
        state_vector = [self @ State([(1,state)]) for state in QN]
        return np.array(state_vector,dtype = float)
    
    #Method that removes components that are smaller than tolerance from the state   
    def remove_small_components(self, tol = 1e-3):
        purged_data = []
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                purged_data.append((amp,basis_state))
        
        return State(purged_data, energy = self.energy)
    
    #Method for ordering states in descending order of amp^2
    def order_by_amp(self):
        data = self.data
        amp_array = np.zeros(len(data))
        
        #Make an numpy array of the amplitudes
        for i,d in enumerate(data):
            amp_array[i] = (data[i][0])**2
        
        #Find ordering of array in descending order
        index = np.argsort(-1*amp_array)
        
        #Reorder data
        reordered_data = data
        reordered_data = [reordered_data[i] for i in index]
        
        return (State(reordered_data))
        
    
    #Method for printing largest component basis states
    def print_largest_components(self,n = 1):
        #Order the state by amplitude
        state = self.order_by_amp()
        
        #Initialize an empty string
        string = ''
        
        for i in range(0,n):
            basis_state = state.data[i][1]
            amp = state.data[i][0]
            if basis_state.isCoupled:
                F = rat(basis_state.F)
                mF = rat(basis_state.mF)
                F1 = rat(basis_state.F1)
                J = rat(basis_state.J)
                I1 = rat(basis_state.I1)
                I2 = rat(basis_state.I2)
#                string =string + ('{:.4f}'.format(N(amp))+' x '+"|F = %s, mF = %s, F1 = %s, J = %s, I1 = %s, I2 = %s>"\
#                      %(F,mF,F1,J,I1,I2))
                string =string + ('{:.4f}'.format(N(amp))+' x '+"|F = %s, mF = %s, F1 = %s, J = %s, I1 = %s, I2 = %s>"\
                      %(F,mF,F1,J,I1,I2))
            elif basis_state.isUncoupled:
                J = rat(basis_state.J)
                mJ = rat(basis_state.mJ)
                I1 = rat(basis_state.I1)
                m1 = rat(basis_state.m1)
                I2 = rat(basis_state.I2)
                m2 = rat(basis_state.m2)
#                string = string + ('{:.4f}'.format(N(amp))+' x '+"|J = %s, mJ = %s, I1 = %s, m1 = %s, I2 = %s, m2 = %s>"\
#                      %(J,mJ,I1,m1,I2,m2))
                string = string + ("|J = %s, mJ = %s, m1 = %s, m2 = %s>"\
                      %(J,mJ,m1,m2))
            
        return string
    
    def find_largest_component(self):
         #Order the state by amplitude
         state = self.order_by_amp()
        
         return state.data[0][1]
    
    #Method for converting the state into the coupled basis
    def transform_to_coupled(self):
        #Loop over the basis states, check if they are already in uncoupled
        #basis and if not convert to coupled basis, output state in new basis
        state_in_coupled_basis = State()
        
        for amp, basis_state in self.data:
            if basis_state.isCoupled:
                state_in_coupled_basis += State((amp,basis_state))
            if basis_state.isUncoupled:
                state_in_coupled_basis += amp*basis_state.transform_to_coupled()
            
        return state_in_coupled_basis
        
