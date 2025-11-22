#!/bin/bash -e
#SBATCH --job-name=train-ours
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH --output=logs/train-ours_%j.out
#SBATCH --error=logs/train-ours_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/slurm/constants.sh"

module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/main.py \
    --multirun \
    model.kwargs.dim_hidden=1000 \
    epochs=20 \
    'wandb.tags=[ours,sparse-fully,tuning]' \
    model.name=fc-baseline-sparse-fully \
    +model.kwargs.sparsity=0.99 \
    +model.kwargs.sparsity_win=0.9 \
    optimizer.weight_decay_j=0.01 \
    optimizer.learning_rate_j=0.02 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.j_d=0.9 \
    optimizer.learning_rate_win=0.3 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.strength_forth=5.0 \
    trainer.fake_dynamics.k=0.5
