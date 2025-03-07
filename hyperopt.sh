#!/bin/bash
#SBATCH -J hyperopt_job
#SBATCH --partition=a100-galvani
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --output=hyperopt-%j.out
#SBATCH --error=hyperopt-%j.err

# 1. Conda korrekt initialisieren
source /mnt/qb/work2/butz1/bst080/miniconda3/etc/profile.d/conda.sh

# 2. Conda-Umgebung mit vollem Pfad aktivieren
conda activate /mnt/qb/work2/butz1/bst080/miniconda3/envs/train_env

# 3. Ergebnisse-Ordner löschen (wichtig für Neustarts)
RESULTS_DIR="/mnt/qb/work/butz/bst080/faceGANtts/hp_results/run_0"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Führe aus OHNE Terminal-Emulation
srun --unbuffered python -u hyperopt.py hyperopt_config.json < /dev/null