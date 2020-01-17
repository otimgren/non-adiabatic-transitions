#!/bin/bash
#SBATCH --partition day
#SBATCH --job-name speed_test
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem-per-cpu 300M
#SBATCH --time 00:30:00
#SBATCH --mail-type all
#SBATCH --mail-user oskari.timgren@yale.edu
#SBATCH --output "C:\Users\Oskari\Google Drive\CeNTREX Oskari\non-adiabatic-transitions/slurm/slurm-%j.out"
#SBATCH --error "C:\Users\Oskari\Google Drive\CeNTREX Oskari\non-adiabatic-transitions/slurm/slurm-%j.out"
module load Python/3.6.4-foss-2018a
python3 /home/fas/demille/omt3/project/C:\Users\Oskari\Google Drive\CeNTREX Oskari\non-adiabatic-transitions options_1_field.json
