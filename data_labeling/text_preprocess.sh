#!/bin/bash

#SBATCH -J text_preprocess
#SBATCH --partition=cpu-galvani   # Use the CPU-only partition
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPUs per task
#SBATCH --mem=50G                # Amount of memory
#SBATCH --time=30-00:00:00       # Maximum runtime
#SBATCH --output=text_preprocess-%j.out
#SBATCH --error=text_preprocess-%j.err

# Diagnostic Phase
# Diagnostic Phase
scontrol show job $SLURM_JOB_ID

# Initialize conda
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate /mnt/qb/home/butz/bst080/miniconda3/envs/label_env

srun conda run -p /mnt/qb/home/butz/bst080/miniconda3/envs/label_env python /home/butz/bst080/facetts-optimizer/data_labeling/txt_preprocess.py --data_path /qb/work2/butz1/bst080/spk_ids_main_weighted_40
