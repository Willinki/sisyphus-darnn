#!/bin/bash -e
#SBATCH --job-name=train-mlp
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=train-mlp_%j.out
#SBATCH --error=train-mlp_%j.err

#PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
#mkdir -p "${PROJECT_ROOT}/logs"
#source "${PROJECT_ROOT}/scripts/slurm/constants.sh"
#
#module load anaconda3/2024.02
#source "$(conda info --base)/etc/profile.d/conda.sh"

#conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/training_mlp.py \
uv run python3.11 scripts/python_scripts/training_perceptron.py \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.sparsity=0.9 \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=9461 \
    model.kwargs.lr=0.0009 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16