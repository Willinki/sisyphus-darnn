#!/bin/bash

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=emnist_clipped_mlp \
    cluster=gpu \
    epochs=100 \
    model.kwargs.optim=sgd \
    'model.kwargs.hidden_dim=128,256,512,1024' \
    'model.kwargs.gain=1.0' \
    'model.kwargs.lr=0.002,0.005,0.01' \
    --multirun