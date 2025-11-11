#!/bin/bash

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=fashionmnist_clipped_mlp \
    cluster=gpu \
    epochs=100 \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024' \
    'model.kwargs.gain=1.0' \
    'model.kwargs.lr=0.001,0.0002,0.0005' \
    'model.kwargs.argmax_margin=1.0,0.8,1.2' \
    --multirun
