#!/bin/bash -e
#SBATCH --job-name=train-hetero
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:59:00
#SBATCH --output=logs/train-hetero_%j.out
#SBATCH --error=logs/train-hetero_%j.err
#SBATCH --comment="preemption=yes;requeue=yes"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/slurm/constants.sh"

module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda
conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/training_mlp_hetero.py \
    --multirun \
    model.kwargs.hidden_dim=100,200,300,400,500,600 \
    model.kwargs.lr=0.003 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    model.prototypes_distro=gaussian \
    model.reset_readout=true \
    data.kwargs.batch_size=16
