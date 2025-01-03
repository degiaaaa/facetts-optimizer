#!/bin/bash
#SBATCH -J facetts_format          # Job name
#SBATCH --partition=cpu-galvani    # Use the CPU-only partition
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPUs per task
#SBATCH --mem=50G                  # Amount of memory
#SBATCH --time=30-00:00:00         # Maximum runtime
#SBATCH --output=facetts-%j.out    # Standard output log
#SBATCH --error=facetts-%j.err     # Standard error log

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID

# Initialize conda
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate /mnt/qb/home/butz/bst080/miniconda3/envs/label_env

# Define the input and output directories
input="/mnt/qb/work2/butz1/bst080/syncnet_output"  # Input is SyncNet output
output="/mnt/qb/work2/butz1/bst080/facetts_input"   # Target FacetTS format

# Run the Python script for FacetTS formatting
srun conda run -p /mnt/qb/home/butz/bst080/miniconda3/envs/label_env python /home/butz/bst080/facetts-optimizer/data_labeling/facetts_fromat.py --input_dir $input --output_dir $output
