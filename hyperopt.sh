#!/bin/bash
#SBATCH -J hyperopt_job
#SBATCH --partition=2080-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --output=hyperopt-%j.out
#SBATCH --error=hyperopt-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de

# Load necessary modules
module load anaconda
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env

# Set cache directory
export CLUSTER_UTILS_CACHE_DIR=./.cache_cluster_utils
rm -rf $CLUSTER_UTILS_CACHE_DIR

# Start hyperparameter optimization
srun python -m cluster_utils.hp_optimization hyperopt_config.json
