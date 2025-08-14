#!/bin/bash
#SBATCH --job-name=final_P3
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=3G
#SBATCH --output=final_P3.out
#SBATCH --error=final_P3.err
    
. ~/.bashrc
conda activate py310

# create output directory
mkdir -p outputfolder_P3
cd outputfolder_P3/

# Loop over different numbers of processors
for n_cpu 
in 1 2 4 8 16 32; do

    echo "Running with $n_cpu CPU(s)..."

    mpirun --oversubscribe -np $n_cpu python ../thermostat_mpi.py $n_cpu
	
    wait

done
