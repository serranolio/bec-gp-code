#!/bin/bash
#SBATCH --partition=kamiak              # Partition (guan, cas, kamiak)
#SBATCH --job-name=detuning_ramp        # Job name
#SBATCH --output=logs/%x_%A_%a.out      # Output file
#SBATCH --error=logs/%x_%A_%a.err       # Error file
#SBATCH --mail-type=ALL                 # Notifications
#SBATCH --mail-user=federico.serrano@wsu.edu
#SBATCH --time=0-03:00:00               # Wall time limit

#SBATCH --array=0-19:1  		# Number of jobs
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=1               # Cores per task
#SBATCH --mem=1G                        # Memory per core

# Initialize the array
list_of_directories=()

for dir in simulation_*/; do
    [ -d "$dir" ] && list_of_directories+=("$dir")
done

len_list_of_directories=${#list_of_directories[@]}

srun python3 "${list_of_directories[${SLURM_ARRAY_TASK_ID}]}"execute_simulation.py
