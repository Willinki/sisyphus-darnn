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
    model.kwargs.dim_hidden=5603 \
    wandb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.25 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=4612 \
    wandb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=3626 \
    wnadb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.0 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=2646 \
    wandb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=1683 \
    wandb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.5 \

uv run python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=760 \
    wandb.experiment='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=3.5 \