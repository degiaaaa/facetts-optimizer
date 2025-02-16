#!/bin/bash
#SBATCH -J train_orig
#SBATCH --partition=a100-galvani #2080-galvani #a100-fat-galvani  #a100-galvani
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=100G                # Amount of memory
#SBATCH --time=3-00:00:00       # Maximum runtime
#SBATCH --output=train_orig-%j.out
#SBATCH --error=train_orig-%j.err
#SBATCH --mail-type=END,FAIL       # Notify when job ends or fails
#SBATCH --mail-user=debie1997@yahoo.de  # Your email address

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID
# Initialize conda
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh
#conda env create --prefix /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env --file /mnt/qb/work/butz/bst080/train_env.yml
# Activate the environment
#conda activate label_env
conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env  #/mnt/qb/home/butz/bst080/miniconda3/envs/label_env

srun conda run -p /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env python /mnt/qb/work/butz/bst080/faceGANtts/run_orig.py
