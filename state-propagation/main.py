# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:32:49 2020

@author: Oskari
"""

from state_propagation_functions import *
import argparse
import json
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions as molecules move to ES lens")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory to options_dict
    options_dict["run_dir"] = args.run_dir
    
    #Generate list of quantum numbers
    QN = generate_QN()
    
    #Get electric and magnetic fields as function of time
    B = get_B_field(options_dict)
    E = get_E_field(options_dict)
    
    #Get hamiltonian as function of E- and B-field
    H = get_Hamiltonian(options_dict)
    
    #Make state vector for lens state
    lens_state_fname = options_dict["state_fname"]
    with open(args.run_dir+lens_state_fname, 'rb') as f:
        lens_state = pickle.load(f)    
    lens_state_vec = lens_state.state_vector(QN)
    
    #Propagate the state in time
    T = options_dict["E_params"]["tau"]*20
    N_steps = int(options_dict["time_params"]["N_steps"])
    probability = propagate_state(lens_state_vec, H, QN, B, E, T, N_steps)
    
    print(probability)
    