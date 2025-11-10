#!/bin/bash
uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=mnist_tanh_mlp \
    cluster=gpu \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024,2048' \
    'model.kwargs.gain=1.0,5.0' \
    'model.kwargs.lr=0.001,0.0002,0.005' \
    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=mnist_binary_mlp \
    cluster=gpu \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024,2048' \
    'model.kwargs.gain=1.0,5.0' \
    'model.kwargs.lr=0.001,0.0002,0.005' \
    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=mnist_clipped_mlp \
    cluster=gpu \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024,2048' \
    'model.kwargs.gain=1.0,5.0' \
    'model.kwargs.lr=0.001,0.0002,0.005' \
    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=mnist_relu_mlp \
    cluster=gpu \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024,2048' \
    'model.kwargs.lr=0.001,0.0002,0.005' \
    --multirun
