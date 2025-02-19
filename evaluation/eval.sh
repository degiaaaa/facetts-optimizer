#!/bin/bash
#SBATCH -J evaluate_tts
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluation-%j.out
#SBATCH --error=evaluation-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de

# Load environment
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh

conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env
export PYTHONPATH="/mnt/qb/work/butz/bst080/faceGANtts:$PYTHONPATH"
srun python /mnt/qb/work/butz/bst080/faceGANtts/evaluation/eval.py 
