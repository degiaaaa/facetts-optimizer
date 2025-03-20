#!/bin/bash
#SBATCH -J reformatdata
#SBATCH --partition=2080-galvani   #cpu-galvani   # Use the CPU-only partition
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=2        # Number of CPUs per task
#SBATCH --mem=30G                # Amount of memory
#SBATCH --time=3-00:00:00       # Maximum runtime
#SBATCH --output=audio-%j.out
#SBATCH --error=audio-%j.err

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID
# Initialize conda
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh

# Activate the environment
#conda activate label_env
conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/label_env  #/mnt/qb/home/butz/bst080/miniconda3/envs/label_env

#srun conda run -p /mnt/qb/work2/butz1/bst080/miniconda3/envs/label_env python /mnt/qb/work/butz/bst080/faceGANtts/data/reformat_data.py 
srun conda run -p /mnt/qb/work2/butz1/bst080/miniconda3/envs/label_env python /mnt/qb/work/butz/bst080/faceGANtts/lrs2_preprocessing/lrs2_split/filelist_split.py
