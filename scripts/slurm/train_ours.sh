#!/bin/bash -e
#SBATCH --job-name=train-ours
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH --output=logs/train-ours_%j.out
#SBATCH --error=logs/train-ours_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/slurm/constants.sh"

module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/main.py \
    --multirun \
    model.kwargs.dim_hidden=1000 \
    epochs=5 \
    'wandb.tags=[ours,emnist]'
