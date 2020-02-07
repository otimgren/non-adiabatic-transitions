# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:32:49 2020

@author: Oskari
"""
import numpy as np
import argparse
import json
import time
import os, sys
import dill



def generate_jobs_files(options_dict):
    """
    This is a function that generates a text file with the command line arguments
    to run jobs using "dead Simple Queue" (dSQ) on the Yale HPC cluster.
    
    input on command line:
    run_dir = path to directory in which the code is run
    options_fname = path to options file that specifies parameters for the jobs
    jobs_fname = path to output file that is used to with dSQ to generate a jobs array
    
    output:
    a text file containing the commands needed to submit jobs to cluster using dSQ
    """
    #Get some parameters from options_dict 
    cluster_params = options_dict["cluster_params"]
    run_dir = options_dict["run_dir"]
    jobs_fname = options_dict["jobs_fname"]
    options_fname = options_dict["options_fname"]
    result_fname = options_dict["result_fname"]
    
    
    #Check how many trajectories there are in the trajectories file
    n_traj_start = options_dict["n_traj_start"]
    n_traj_end = options_dict["n_traj_end"]
    n_trajectories = n_traj_end - n_traj_start

    #Max size for jobsarray is 10000 jobs so if n_trajectories > 1e4, need
    #to split the job into multiple arrays
    n_max = int(1e4)
    n_loops = int(n_trajectories/n_max)+1
    
    jobs_files = []
    
    #Loop over number of jobsfiles needed
    for n in range(0, n_loops):
        
        #Figure out what trajectory indices to use
        if n < n_loops-1:
            i_start = n*n_max
            i_end = (n+1)*n_max
        elif n == n_loops-1:
            i_start = n*n_max
            i_end = n_traj_end
        
        #Append name of jobsfile to list of jobsfile names
        jobs_files.append(jobs_fname+'_'+str(n)+'.txt')
        #Open the text file that is used for the jobsfile
        with open(run_dir + '/jobs_files/' + jobs_fname+'_'+str(n)+'.txt', 'w+') as f:
            #Loop over trajectories
            for i in range(i_start,i_end):
                
                #Start printing into the jobs file 
                #Load the correct modules
                print("module load miniconda", file=f, end = '; ')
                print("source deactivate", file=f, end = '; ')
                print("source activate non_adiabatic", file=f, end = '; ')
                
                #Generate the string that executes the program and gives it parameters
                exec_str =  ("python " + cluster_params["prog"] + " "
                                + run_dir + " " + options_fname + " " + result_fname
                                + " {} ".format(i))
                # if i == 0:
                #     exec_str += "--save_fields"
                    
                print(exec_str, file=f)
    
    #Also initialize the results file
    with open(run_dir + '/results/' + result_fname, 'w+') as f:
        print("Options:", file = f)
        print(options_dict, file = f)
        
        print(20*'*', file = f)
        
        #Print headers for the results
        headers = ['Probability','Trajectory', 'Error']
        headers_str = '\t\t'.join(headers)
        print(headers_str, file = f)
        
    #Return the list of jobsfile names
    return jobs_files


def generate_batchfiles(jobs_files, options_dict):
    """
    Function that generates batchfiles based on given jobs files using dSQ
    """
    #Settings for dSQ
    cluster_params = options_dict["cluster_params"]
    memory_per_cpu = cluster_params["mem-per-cpu"]
    time = cluster_params["time"]
    mail_type = cluster_params["mail-type"]
    
    #Setting paths
    run_dir = options_dict['run_dir']
    
    #Initialize container for batchfiles
    batchfile_paths = []
    
    #Loop over jobsfiles
    for jobs_fname in jobs_files:
        jobs_path =run_dir + '/jobs_files/' + jobs_fname
        batchfile_path = run_dir + '/slurm/' + 'dsq-' +jobs_fname
        batchfile_paths.append(batchfile_path)
        
        #Generate the string to execute
        exec_str = ('dsq --job-file ' + jobs_path + ' --mem-per-cpu ' + memory_per_cpu
                    +' -t ' + time + ' --mail-type '+ mail_type + ' -o /dev/null --batch-file '
                    + batchfile_path)
        
        os.system(exec_str)
        
        #Write which partition to use to the file on line 2
        with open(batchfile_path, 'r') as f:
            lines = f.readlines()
        
        text = ('#SBATCH --partition '+options_dict["cluster_params"]["partition"])
        if options_dict["cluster_params"]["requeue"]:
            text += '\n#SBATCH --requeue\n'
        lines.insert(1, text)
        
        with open(batchfile_path, 'w') as f:
            f.writelines(lines)
    
    #Return the path to the batchfile
    return batchfile_paths

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
    #Get arguments from command line and parse them
    parser = argparse.ArgumentParser(description="A script for studying non-adiabatic transitions")
    parser.add_argument("run_dir", help = "Run directory")
    parser.add_argument("options_fname", help = "Filename of the options file")
    parser.add_argument("result_fname", help = "Filename for storing results")
    parser.add_argument("jobs_fname", help = "Filename for storing jobs")
    parser.add_argument("--jobs", help="If true, generate jobsfile"
                    , action = "store_true")
    parser.add_argument("--batch", help="If true, generate jobsfile and batchfile from that"
                    , action = "store_true")
    parser.add_argument("--submit", help="If true, generate batchfile and submit to cluster"
                        , action = "store_true")
    
    args = parser.parse_args()
    
    #Load options file
    with open(args.run_dir + '/options/' + args.options_fname) as options_file:
        options_dict = json.load(options_file)
    
    #Add run directory and options filename to options_dict
    options_dict["run_dir"] = args.run_dir
    options_dict["options_fname"] = args.options_fname
    options_dict["result_fname"] = args.result_fname + time.strftime('_%Y-%m-%d_%H-%M-%S') +'.txt'
    options_dict["jobs_fname"] = args.jobs_fname
    
    #Different uses of main.py. Can either make a job file, job file + batch file,
    #or make both files and submit
    if args.jobs:
        generate_jobs_files(options_dict)
        
    elif args.batch:
        generate_jobs_files(options_dict)
        generate_batchfiles(options_dict)
    
    #Generate a jobs file and batch file, and submit to cluster
    elif args.submit:
        jobs_files = generate_jobs_files(options_dict)
        batchfile_paths = generate_batchfiles(jobs_files, options_dict)
        for batchfile_path in batchfile_paths:
            os.system("sbatch {}".format(batchfile_path))
        