# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:14:50 2020

@author: Oskari

This is a program that generates a text file with the command line arguments
to run jobs using "dead Simple Queue" (dSQ) on the Yale HPC cluster.

input on command line:
run_dir = path to directory in which the code is run
options_fname = path to options file that specifies parameters for the jobs
jobs_fname = path to output file that is used to with dSQ to generate a jobs array


output:
a text file containing the commands needed to submit jobs to cluster using dSQ

"""
import argparse
import numpy as np
import json

def generate_field_param_array(param_dict):
    """
    Function that generates an array of values for a scan over a field parameter,
    e.g. z-component of electric field
    
    input:
    param_dict =  dictionary that specifies if parameter is to be scanned, what
    value the parameter should take etc.
    
    return:
    an array that contains the parameter
    """
    #Check if the parameter is supposed to be scanned
    scan = param_dict["scan"]
    
    #Two cases: parameter is scanned or not scanned
    if scan:
        #If parameter is scanned, find the parameters for the scan
        p_ini = param_dict["min"]
        p_fin = param_dict["max"]
        N = param_dict["N"]
        param = np.linspace(p_ini, p_fin,N)
        
    else:
        #If not scanned, set value
        value = param_dict["value"]
        param = np.array(value)
        
    #If the unit is specified, also get that
    try:
        unit = param_dict["unit"]
    except ValueError:
        pass
    
    return param,unit
    

if __name__ == "__main__":
    #Get name of options file and output file from commandline
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("jobs_fname", help = "Filename for storing jobs")
    parser.add_argument("results_fname", help = "Filename for storing results")
    parser.add_argument("--submit", help="If true, generate batchfile and submit to cluster"
                        , action = "store_true")
    args = parser.parse_args()
    
    #Get options dict from file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    cluster_params = options_dict["cluster_params"]
    
    #Generate arrays for field parameters
    param_names = []
    units = []
    for param_name, param_dict in options_dict["field_params"].items():
        param, unit = generate_field_param_array(param_dict)
        exec(param_name+'= param')
        param_names.append(param_name)
        units.append(unit)
        
    #Generate a table of the field parameters to use for each job
    mesh_params = ','.join([param_name for param_name in param_names])
    exec('array_list = np.meshgrid('+mesh_params+')')
    flattened_list = []
    for array in array_list:
        flattened_list.append(array.flatten())
    field_param_table = np.vstack(flattened_list).T
    
    #Open the text file that is used for the jobsfile
    with open(args.run_dir + '/jobs_files/' + args.jobs_fname, 'w+') as f:
        #Loop over rows of the field parameter table
        for row in field_param_table:
            #Extract the parameters for this job
            for i, param_value in enumerate(row):
                param_name = param_names[i]
                exec(param_name +'= param_value')
        
            #Start printing into the jobs file 
            #Load the correct modules
            print("module load miniconda", file=f, end = '; ')
            print("source deactivate", file=f, end = '; ')
            print("source activate non_adiabatic", file=f, end = '; ')
            
            #Generate the string that executes the program and gives it parameters
            exec_str =  ("python " + cluster_params["prog"] + " "
                            + args.run_dir + " " + args.options_fname + " " + args.jobs_fname
                            + " {} {} {} {} {} {} {} {}".format(Ex0,Ey0,Ez0,tau_E,
                                Bx0,By0,Bz0,f_B))
            print(exec_str, file=f)
    
    #Also initialize the results file
    with open(args.run_dir + '/results/' + args.results_fname, 'w+') as f:
        print("Time dependence of E-field:", file = f)
        print(options_dict["E_t"], file = f)
        
        print("Time dependence of B-field:", file = f)
        print(options_dict["B_t"]+"\n\n" + 10*'*' +'\n', file = f)
        
        #Print headers for the results
        headers = ['Probability']
        for param_name, unit in zip(param_names,units):
            headers.append(param_name +'/'+unit)
        headers_str = '\t\t'.join(headers)
        print(headers_str, file = f)
        
        