#!/bin/bash -e
#SBATCH --job-name=train-ours
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -t 01:59:00
#SBATCH --output=logs/train-ours_%j.out
#SBATCH --error=logs/train-ours_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

#PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
#mkdir -p "${PROJECT_ROOT}/logs"
#source "${PROJECT_ROOT}/scripts/slurm/constants.sh"
#
#module load anaconda3/2024.02
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
# conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/main.py \
uv run python3.11 scripts/python_scripts/main.py \
    --multirun \
    model.kwargs.dim_hidden=320 \
    epochs=20 \
    'wandb.tags=[ours,sparsity]' \
    model.name=fc-baseline-sparse \
    +model.kwargs.sparsity=0.9 \
    optimizer.weight_decay_j=0.02 \
    optimizer.learning_rate_j=0.01 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.j_d=0.9 \
    optimizer.learning_rate_win=0.05 \
    model.kwargs.threshold_in=1.4 \
    model.kwargs.strength_forth=4.0
# parametri scelti in base a H=100, sparsity=0.9