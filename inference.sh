#!/bin/bash
#SBATCH -J inference_tts
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=inference-%j.out
#SBATCH --error=inference-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de

# Load environment
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env

# Set CUDA memory allocation to avoid fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Show GPU info
nvidia-smi

# Run inference script (No extra arguments needed)
srun python /mnt/qb/work/butz/bst080/faceGANtts/inference.py
