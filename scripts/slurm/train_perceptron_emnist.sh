#!/bin/bash -e
#SBATCH --job-name=train-perc
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=train-perc_%j.out
#SBATCH --error=train-perc_%j.err

#PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
#mkdir -p "${PROJECT_ROOT}/logs"
#source "${PROJECT_ROOT}/scripts/slurm/constants.sh"
#
#module load anaconda3/2024.02
#source "$(conda info --base)/etc/profile.d/conda.sh"

#conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/training_mlp.py \
uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=1050 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=3100 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=6150 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=10200 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=15250 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=21300 \
    model.kwargs.lr=0.0009 \
    model.kwargs.sparsity=0.9 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project="emnist_scaling" \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    data.kwargs.batch_size=16
