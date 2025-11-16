#!/bin/bash -e
#SBATCH --job-name=train-ours
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:59:00
#SBATCH --output=logs/train-ours_%j.out
#SBATCH --error=logs/train-ours_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/slurm/constants.sh"

module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
# conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/composable_experiment/main.py \
#     --multirun \
#     model.kwargs.dim_hidden=100 \
#     optimizer.learning_rate_win=0.115,0.1,0.085,0.07,0.055 \
#     optimizer.weight_decay_win=0.0,0.005 \
#     model.kwargs.strength_back=0.9,1.4,1.9,2.4,2.9 \
#     epochs=25
conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/composable_experiment/main.py \
    --multirun \
    model.kwargs.dim_hidden=100 \
    optimizer.learning_rate_j=0.003 \
    optimizer.weight_decay_j=0.0,0.005 \
    model.kwargs.threshold_j=1.4,1.9,2.4 \
    epochs=50 \
    'wandb.tags=[tuning,ours,simple-hparams]'