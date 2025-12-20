#!/bin/bash
#SBATCH --job-name=train-ours
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=train-emnist_%j.out
#SBATCH --error=train-emnist_%j.err


# Conda
uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=6000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.25 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=5000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=4000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.0 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=3000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=2000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.5 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=1000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=3.5 \