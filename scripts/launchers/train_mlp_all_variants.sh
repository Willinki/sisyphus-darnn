#!/bin/bash
uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=cifar10_features_s_tanh_mlp \
    cluster=gpu \
    'model.kwargs.hidden_dim=100,200,400,800,1600' \
    'model.kwargs.gain=1.0,5.0' \
    'data.kwargs.x_transform=identity,sign' \
    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=cifar10_features_s_binary_mlp \
    cluster=gpu \
    'model.kwargs.hidden_dim=100,200,400,800,1600' \
    'model.kwargs.gain=1.0,5.0' \
    'data.kwargs.x_transform=identity,sign' \
    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=cifar10_features_s_clipped_mlp \
    cluster=gpu \
    'model.kwargs.hidden_dim=100,200,400,800,1600' \
    'model.kwargs.gain=1.0,5.0' \
    'data.kwargs.x_transform=identity,sign' \
    --multirun
