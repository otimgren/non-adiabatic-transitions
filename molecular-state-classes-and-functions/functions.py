# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:39:49 2019

Storing functions needed for the notebook in the same folder in this file.

@author: Oskari
"""

import munkres
import numpy as np
from classes import UncoupledBasisState, State
from tqdm import tqdm
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

#Function for keeping track eigenstates when repeatedly diagonalizing a matrix
def eigenshuffle(Asequence):
    """
    Consistent sorting for an eigenvalue/vector sequence
    
    Based on eigenshuffle.m 3.0 (2/18/2009) for MATLAB by John D'Errico
    http://www.mathworks.com/matlabcentral/fileexchange/22885
    
    Python adaptation by Brecht Machiels
        <brecht.machiels@esat.kuleuven.be>
    
    Requires NumPy (http://numpy.scipy.org)
    and munkres.py by Brian M. Clapper
    (http://www.clapper.org/software/python/munkres/)
    
    Parameters
    ----------
    Asequence : ndarray, shape (N, M, M)
        An array of eigenvalue problems. If Asequence is a 3-d numeric array, 
        then each plane of Asequence must contain a square matrix that will be 
        used to call numpy.linalg.eig.
        
        numpy.linalg.eig will be called on each of these matrices to produce a 
        series of eigenvalues/vectors, one such set for each eigenvalue problem.
        
    Returns
    -------
    Dseq : ndarray, shape (M,)
        A pxn array of eigen values, sorted in order to be consistent with each 
        other and with the eigenvectors in Vseq.
    Vseq : ndarray, shape (M, M)
        A 3-d array (pxpxn) of eigenvectors. Each plane of the array will be 
        sorted into a consistent order with the other eigenvalue problems. The 
        ordering chosen will be one that maximizes the energy of the consecutive
        eigensystems relative to each other.
        
    See Also
    --------
    numpy.linalg.eig
    
    Example
    -------
    >>> import numpy as np
    >>> from nport.eigenshuffle import eigenshuffle
    >>>
    >>> np.set_printoptions(precision=5, suppress=True)
    >>>
    >>> def Efun(t):
    >>>     return np.array([
    >>>         [1,     2*t+1 , t**2 ,   t**3],
    >>>         [2*t+1, 2-t   , t**2 , 1-t**3],
    >>>         [t**2 , t**2  , 3-2*t,   t**2],
    >>>         [t**3 , 1-t**3, t**2 ,  4-3*t]])
    >>> 
    >>> Aseq = np.zeros( (21, 4, 4) )
    >>> tseq = np.arange(-1, 1.1, 0.1)
    >>> for i, t in enumerate(tseq):
    >>>     Aseq[i] = Efun(t)
    >>>
    >>> [Dseq, Vseq] = eigenshuffle(Aseq)
        
    To see that eigenshuffle has done its work correctly, look at the
    eigenvalues in sequence, after the shuffle.
    
    >>> print np.hstack([np.asarray([tseq]).T, Dseq]).astype(float)
        
    [[-1.      8.4535  5.      2.3447  0.2018]
     [-0.9     7.8121  4.7687  2.3728  0.4464]
     [-0.8     7.2481  4.56    2.3413  0.6505]
     [-0.7     6.7524  4.3648  2.2709  0.8118]
     [-0.6     6.3156  4.1751  2.1857  0.9236]
     [-0.5     5.9283  3.9855  2.1118  0.9744]
     [-0.4     5.5816  3.7931  2.0727  0.9525]
     [-0.3     5.2676  3.5976  2.0768  0.858 ]
     [-0.2     4.9791  3.3995  2.1156  0.7058]
     [-0.1     4.7109  3.2     2.1742  0.5149]
     [-0.      4.4605  3.      2.2391  0.3004]
     [ 0.1     4.2302  2.8     2.2971  0.0727]
     [ 0.2     4.0303  2.5997  2.3303 -0.1603]
     [ 0.3     3.8817  2.4047  2.3064 -0.3927]
     [ 0.4     3.8108  2.1464  2.2628 -0.62  ]
     [ 0.5     3.8302  1.8986  2.1111 -0.8399]
     [ 0.6     3.9301  1.5937  1.9298 -1.0537]
     [ 0.7     4.0927  1.2308  1.745  -1.2685]
     [ 0.8     4.3042  0.8252  1.5729 -1.5023]
     [ 0.9     4.5572  0.4039  1.4272 -1.7883]
     [ 1.      4.8482  0.      1.3273 -2.1755]]
    Here, the columns are the shuffled eigenvalues. See that the second
    eigenvalue goes to zero, but the third eigenvalue remains positive. We can
    plot eigenvalues and see that they have crossed, near t = 0.35 in Efun.
    >>> from pylab import plot, show
    >>> plot(tseq, Dseq); show()
    For a better appreciation of what eigenshuffle did, compare the result of
    numpy.linalg.eig directly on Efun(0.3) and Efun(0.4). Thus:
    
    >>> [D3, V3] = np.linalg.eig(Efun(0.3))
    >>> print V3
    >>> print D3
    [[ 0.74139 -0.3302   0.53464 -0.23551]
     [-0.64781 -0.57659  0.4706  -0.16256]
     [-0.00865 -0.10006 -0.44236 -0.89119]
     [ 0.17496 -0.74061 -0.54498  0.35197]]
    [-0.39272  3.88171  2.30636  2.40466]
    >>> [D4, V4] = np.linalg.eig(Efun(0.4))
    >>> print V4
    >>> print D4
    [[ 0.73026 -0.42459  0.49743  0.19752]
     [-0.66202 -0.62567  0.35297  0.21373]
     [-0.01341 -0.16717  0.25513 -0.95225]
     [ 0.16815 -0.63271 -0.75026 -0.09231]]
    [-0.62001  3.8108   2.2628   2.14641]
    With no sort or shuffle applied, look at V3[2]. See that it is really
    closest to V4[1], but with a sign flip. Since the signs on the
    eigenvectors are arbitrary, the sign is changed, and the most consistent
    sequence will be chosen. By way of comparison, see how the eigenvectors in
    Vseq have been shuffled, the signs swapped appropriately.
    >>> print Vseq[13, :, :].astype(float)
    [[ 0.3302   0.23551 -0.53464  0.74139]
     [ 0.57659  0.16256 -0.4706  -0.64781]
     [ 0.10006  0.89119  0.44236 -0.00865]
     [ 0.74061 -0.35197  0.54498  0.17496]]
    >>> print Vseq[14, :, :].astype(float)
    [[ 0.42459 -0.19752 -0.49743  0.73026]
     [ 0.62567 -0.21373 -0.35297 -0.66202]
     [ 0.16717  0.95225 -0.25513 -0.01341]
     [ 0.63271  0.09231  0.75026  0.16815]]
    """
    # alternative implementations:
    #  * http://www.mathworks.com/matlabcentral/fileexchange/29463-eigenshuffle2
    #  * http://www.mathworks.com/matlabcentral/fileexchange/29464-rootshuffle-m ?
    
    # Is Asequence a 3-d array?
    Ashape = np.shape(Asequence)
    if Ashape[-1] != Ashape[-2]:
        print("Asequence must be a (nxpxp) array of "
                          "eigen-problems, each of size pxp")
        raise Exception
        
    p = Ashape[-1]
    if len(Ashape) < 3:
        n = 1
        Asequence = np.asarray([Asequence], dtype=complex)
    else:
        n = Ashape[0]

    # the initial eigenvalues/vectors in nominal order
    Vseq = np.zeros( (n, p, p), dtype=complex )
    Dseq = np.zeros( (n, p), dtype=complex )

    for i in range(n):
        D, V = np.linalg.eig( Asequence[i] )
        # initial ordering is purely in decreasing order.
        # If any are complex, the sort is in terms of the
        # real part.
        tags = np.argsort(D.real, axis=0)[::-1]
        
        Dseq[i] = D[:, tags]
        Vseq[i] = V[:, tags]
    
    # now, treat each eigenproblem in sequence (after the first one.)
    m = munkres.Munkres()
    for i in range(1, n):
        # compute distance between systems
        D1 = Dseq[i - 1]
        D2 = Dseq[i]
        V1 = Vseq[i - 1]
        V2 = Vseq[i]
        dist = ((1 - np.abs(np.dot(np.transpose(V1), V2))) *
                np.sqrt(distancematrix(D1.real, D2.real)**2 +
                        distancematrix(D1.imag, D2.imag)**2))

        # Is there a best permutation? use munkres.
        reorder = m.compute(np.transpose(dist))
        reorder = [coord[1] for coord in reorder]

        Vs = Vseq[i]
        Vseq[i] = Vseq[i][:, reorder]
        Dseq[i] = Dseq[i, reorder]

        # also ensure the signs of each eigenvector pair
        # were consistent if possible
        S = np.squeeze( np.sum( Vseq[i - 1] * Vseq[i], 0 ).real ) < 0
       
        Vseq[i] = Vseq[i] * (-S * 2 - 1)

    return Dseq, Vseq

#function to sort the output of eig by using the Hungarian algorithm
def sort_evals(V1, D1, V2, D2):
    #calculate 'distance' between current system and sorting system
    #dist = ((1 - np.abs(np.dot(V1.conj().T, V2)))  *
    #           (np.sqrt(distancematrix(D1.real, D2.real)**2 +
    #           distancematrix(D1.imag, D2.imag)**2)))
    dist = ((1 - np.absolute(np.dot(V1.conj().T, V2))))
    #get reordering of state vec by using function 'munkres.m' (I don't
    #understand how this works, but it does the job...)
    m = munkres.Munkres()
    #reorder = m.compute(np.transpose(dist))
    reorder = m.compute(dist)
    reorder = np.array([coord[1] for coord in reorder])
    #reorder state vecs and evals based on what munkres gives
    V_sorted = V1[:,reorder]
    D_sorted = D1[reorder]
    return V_sorted, D_sorted

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