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
import dill
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
    parser.add_argument("vz", type = float)
    parser.add_argument("--save_fields", help = "If true, save the E-and B-fields",
                                                 action = "store_true")

    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    result_fname =  args.result_fname  
    save_fields = args.save_fields
      
    #Generate list of quantum numbers
    QN = generate_QN()
    
    #Define the trajectory
    def molecule_position(t, r0, v):
        """
        Functions that returns position of molecule at a given time for given initial position and velocity.
        inputs:
        t = time in seconds
        r0 = position of molecule at t = 0 in meters
        v = velocity of molecule in meters per second
        
        returns:
        r = position of molecule in metres
        """
        #If t is iterable 
        try:
            r =  np.array([r0 + v*t_i for t_i in t])
        except TypeError: 
            r = r0 + v*t
        
        return r

    #Define the position of the molecule as a fucntion of time
    r0 = np.array(options_dict["r0"])
    vz = args.vz
    v = np.array((0,0,vz))
    x_t = lambda t: molecule_position(t, r0, v)

    #Define the total time for which the molecule is simulated
    z0 = r0[2]
    L = 1.1
    z1 = z0 + L
    T = np.abs(L/vz)
    
    #Get electric and magnetic fields as function of position
    B = get_B_field(options_dict)
    E = get_E_field(options_dict)
    
    #Get B and E as functions of time:
    E_t = lambda t: E(x_t(t))
    B_t = lambda t: B(x_t(t))*0
    
    #Get hamiltonian as function of E- and B-field
    H = get_Hamiltonian(options_dict)
    
    #Make state vector for lens state
    lens_state_fname = options_dict["state_fname"]
    with open(args.run_dir+lens_state_fname, 'rb') as f:
        lens_state = pickle.load(f)    
    lens_state_vec = lens_state.state_vector(QN)
    
    #Get number of timsteps to take from options dictionary
    N_steps = int(options_dict["time_params"]["N_steps"])
    
    #Propagate the state in time
    results_dict = propagate_state(lens_state_vec, H, QN, B_t, E_t, T, N_steps, save_fields=save_fields)
    probability = results_dict["probability"]
    
    #If fields are saved, pickle them
    if save_fields:
        E_field = results_dict["E_field"]
        B_field = results_dict["B_field"]
        t_array = results_dict["t_array"]
        
        with open(args.run_dir+'/fields/E_field_'+result_fname[:-3]+'pickle', 'wb+') as f:
            dill.dump(E_field,f)
            
        with open(args.run_dir+'/fields/B_field_'+result_fname[:-3]+'pickle', 'wb+') as f:
            dill.dump(B_field,f)
            
        with open(args.run_dir+'/fields/t_array_'+result_fname[:-3]+'pickle', 'wb+') as f:
            dill.dump(t_array,f)
    
    
    #Append results into file
    with open(args.run_dir + '/results/' + args.result_fname, 'a') as f:
        results_list = ["{:.7e}".format(probability[0,0]), str(vz), str(probability > 1)]
        results_str = "\t\t".join(results_list)
        print(results_str, file = f)

    