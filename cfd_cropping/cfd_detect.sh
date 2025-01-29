#!/bin/bash
#SBATCH -J face_detection
#SBATCH --partition=cpu-galvani   # Use the CPU-only partition
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPUs per task
#SBATCH --mem=50G                # Amount of memory
#SBATCH --time=30-00:00:00       # Maximum runtime
#SBATCH --output= face_detection-%j.out
#SBATCH --error= face_detection-%j.err

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID
# Initialize conda
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh

# Activate the environment
#conda activate label_env
conda activate /mnt/qb/home/butz/bst080/miniconda3/envs/label_env

srun conda run -p /mnt/qb/home/butz/bst080/miniconda3/envs/label_env python /mnt/qb/work/butz/bst080/faceGANtts/cfd_cropping/cfd_detect.py --data_path /qb/work2/butz1/bst080/working_copy_main
