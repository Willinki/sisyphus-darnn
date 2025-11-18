#!/bin/bash -e
#SBATCH --job-name=train-mlp
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:59:00
#SBATCH --output=logs/train-mlp_%j.out
#SBATCH --error=logs/train-mlp_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/slurm/constants.sh"

module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/training_mlp.py \
    --multirun \
    experiment=entangled_mnist_binary_mlp \
    model.kwargs.lr=0.005 \
    epochs=20 \
    'wandb.tags=[mlp]' \
    model.kwargs.loss_type="argmax_margin"
