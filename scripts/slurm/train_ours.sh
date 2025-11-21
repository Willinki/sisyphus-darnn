#!/bin/bash -e
#SBATCH --job-name=train-ours
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:59:00
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
    model.kwargs.dim_hidden=100 \
    epochs=50 \
    'wandb.tags=[ours,jd]' \
    trainer.kwargs.momentum=0.0 \
    model.kwargs.j_d=0.7 \
    trainer.fake_dynamics.enabled=true \
    trainer.gating.enabled=false \
    trainer.gating.warmup_epochs=0 \
    trainer.gating.shift=1.0,1.25,0.75