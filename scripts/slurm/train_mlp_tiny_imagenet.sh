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
uv run python3.11 scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    model.name="clipped-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=600 \
    'model.kwargs.lr=0.0005,0.001,0.0001,0.002,0.005' \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16
