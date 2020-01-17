# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:32:49 2020

@author: Oskari
"""

from state_propagation_functions import *
import argparse
import json
import pickle
import time
import os, sys

def generate_batchfile(options_dict):
    """
    Function that generates a batchfile that is used to submit the scan to 
    a HPC cluster using slurm
    """
    run_dir = options_dict["run_dir"]
    options_fname = options_dict["options_fname"]
    batch_fname = run_dir + '/slurm/' + options_dict["batch_fname"]
    cluster_params = options_dict["cluster_params"]
    
    #Open file and write job submission options to it
    with open(batch_fname,'w') as f:
        print("#!/bin/bash", file = f)
        if cluster_params["requeue"]:
           print("#SBATCH --requeue", file=f)
        
        print("#SBATCH --partition "       + cluster_params["partition"],        file=f)
        print("#SBATCH --job-name "        + cluster_params["job-name"],         file=f)
        print("#SBATCH --ntasks "          + cluster_params["ntasks"],           file=f)
        print("#SBATCH --cpus-per-task "   + cluster_params["cpus-per-task"],    file=f)
        print("#SBATCH --mem-per-cpu "     + cluster_params["mem-per-cpu"],      file=f)
        print("#SBATCH --time "            + cluster_params["time"],             file=f)
        print("#SBATCH --mail-type "       + cluster_params["mail-type"],        file=f)
        print("#SBATCH --mail-user "       + cluster_params["mail-user"],        file=f)
        print("#SBATCH --output \""        + run_dir+"/slurm/slurm-%j.out"+"\"", file=f)
        print("#SBATCH --error \""         + run_dir+"/slurm/slurm-%j.out"+"\"", file=f)
        print("\nmodule load miniconda\n", file=f)
        print("source activate non_adiabatic\n", file=f)
        exec_str =  ("python3 " + cluster_params["prog"] + " "
                        + run_dir + " " + options_fname)
        print(exec_str, file=f)
    print(f"Generated batch file: {batch_fname}")
    return batch_fname
        

def run_scan(options_dict):
    #Record start time
    start = time.time()
    
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
    
    #Record end time
    end = time.time()
    
    time_elapsed = end-start
    
    return probability, time_elapsed
    

if __name__ == "__main__":
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("--submit", help="If true, generate batchfile and submit to cluster"
                        , action = "store_true")
    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
        
    #Generate a batch file and submit to cluster
    if args.submit:
        batch_fname = generate_batchfile(options_dict)
        os.system(f"sbatch {batch_fname}")
        
    #If this program isn't used for submitting, it is used to run the scan
    else: 
        probability, time_elapsed = run_scan(options_dict)
        
        with open(args.run_dir+'/results/'+args.result_fname,'w') as f:
            print("P = {:.5f}, time_elapsed = {:.2f}".format(probability[0][0], time_elapsed)
            , file  = f)