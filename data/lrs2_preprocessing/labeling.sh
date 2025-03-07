#!/bin/bash
#SBATCH -J labelingpretrain
#SBATCH --partition=cpu-galvani   # Use the CPU-only partition
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPUs per task
#SBATCH --mem=50G                # Amount of memory
#SBATCH --time=30-00:00:00       # Maximum runtime
#SBATCH --output=labeling_pretrain-%j.out
#SBATCH --error=labeling_pretrain-%j.err

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID
# Initialize conda
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh

srun conda run -p /mnt/qb/work2/butz1/bst080/miniconda3/envs/label_env python /mnt/qb/work/butz/bst080/faceGANtts/data/lrs2_preprocessing/labeling.py 
#srun conda run -p /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env python /mnt/qb/work/butz/bst080/faceGANtts/data/lrs2_preprocessing/count_speaker.py
