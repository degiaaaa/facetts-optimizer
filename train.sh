#!/bin/bash
#SBATCH -J DiscAmpScaler_genScheduler_gradclips
#SBATCH --partition=a100-galvani  #2080-galvani #a100-fat-galvani  #a100-galvani
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=50G                # Amount of memory
#SBATCH --time=3-00:00:00       # Maximum runtime
#SBATCH --output=DiscAmpScaler_genScheduler_gradclips_%j.out                     #_gan_freeze5_warmup2_lr4_micro16_den0_adv01
#SBATCH --error=DiscAmpScaler_genScheduler_gradclips_%j.err
#SBATCH --mail-type=END,FAIL       # Notify when job ends or fails
#SBATCH --mail-user=debie1997@yahoo.de  # Your email address

# Diagnostic Phase
scontrol show job $SLURM_JOB_ID
# Initialize conda
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh

conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env  #/mnt/qb/home/butz/bst080/miniconda3/envs/label_env

#nvidia-smi 
#srun python /mnt/qb/work/butz/bst080/faceGANtts/migrate_checkpoint.py # nur für cpu
srun python /mnt/qb/work/butz/bst080/faceGANtts/train.py
