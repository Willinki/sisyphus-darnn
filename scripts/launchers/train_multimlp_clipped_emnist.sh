#!/bin/bash

#uv run python3.11 scripts/python_scripts/training_mlp.py \
#    experiment=emnist_multi_mlp.yaml \
#    cluster=gpu \
#    epochs=100 \
#    model.kwargs.optim=sgd \
#    model.kwargs.hidden_dim=128 \
#    'model.kwargs.kappa=1.0,1.5,0.5' \
#    'model.kwargs.lr=0.002,0.005,0.01' \
#    --multirun

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=emnist_multi_mlp.yaml \
    cluster=gpu \
    epochs=100 \
    model.kwargs.optim=sgd \
    model.kwargs.hidden_dim=512 \
    model.kwargs.kappa=3.0 \
    model.kwargs.lr=0.0005