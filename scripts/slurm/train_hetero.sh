#!/bin/bash -e
#SBATCH --job-name=train-mlp
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=train-mlp_%j.out
#SBATCH --error=train-mlp_%j.err

# Conda
uv run python3.11 scripts/python_scripts/training_mlp_hetero.py \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.hidden_dim=100 \
    model.kwargs.input_dim=784 \
    model.kwargs.lr=0.003 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.clamp=true \
    model.kwargs.optim=sgd \
    model.kwargs.loss_type=cross_entropy \
    model.kwargs.num_hidden_layers=2 \
    model.prototypes_distro=gaussian \
    model.reset_readout=true \
    data.kwargs.batch_size=16
