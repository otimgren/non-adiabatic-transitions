# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:53:13 2020

@author: Oskari

Program that is called by the batch file to run simulation for 
a given set of parameters
"""

from state_propagation_functions import *
from datetime import datetime
import pickle
import argparse
import numpy as np
import json

if __name__ == "__main__":
    #Get path info and electric field parameters from command line
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("Ex0", type = float)
    parser.add_argument("Ey0", type = float)
    parser.add_argument("Ez0", type = float)
    parser.add_argument("vz", type = float)
    parser.add_argument("Bx0", type = float)
    parser.add_argument("By0", type = float)
    parser.add_argument("Bz0", type = float)
    parser.add_argument("f_B", type = float)
    
    args = parser.parse_args()
    
    #Load options dict
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    result_fname =  args.result_fname  
    
    #Make arrays of E- and B-field parameters
    E0 = np.array((args.Ex0,args.Ey0,args.Ez0))
    B0 = np.array((args.Bx0,args.By0,args.Bz0))
    l_lens = 20e-3 #Rise time E-field of lens electric field
    vz = args.vz
    tau_E = l_lens/vz
    f_B = args.f_B
    
    #Generate list of quantum numbers
    QN = generate_QN()
    
    #Get electric and magnetic fields as function of time
    B = get_B_field(B0, f_B, tau_E, options_dict)
    E = get_E_field(E0, tau_E, options_dict)
    
    #Get hamiltonian as function of E- and B-field
    H = get_Hamiltonian(options_dict)
    
    #Make state vector for lens state
    lens_state_fname = options_dict["state_fname"]
    with open(args.run_dir+lens_state_fname, 'rb') as f:
        lens_state = pickle.load(f)    
    lens_state_vec = lens_state.state_vector(QN)
    
    #Propagate the state in time
    T = tau_E*20
    probability = propagate_state(lens_state_vec, H, QN, B, E, T)
    
    #Append results into file
    with open(args.run_dir + '/results/' + args.result_fname, 'a') as f:
        results_list = ["{:.7e}".format(value) for value in [probability[0,0], E0[0], 
                        E0[1], E0[2], vz, B0[0], B0[1], B0[2], f_B]]
        results_str = "\t\t".join(results_list)
        print(results_str, file = f)

    