#!/bin/bash

uv run python3.11 scripts/python_scripts/main.py \
    --config-name=ours_sparse_mnist \
    model.kwargs.dim_hidden=1000 \
    epochs=7 \
    model.kwargs.j_d=0.7 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.90 \
    'model.kwargs.strength_forth=5.0,4.0,3.5,3.0' \
    'model.kwargs.strength_back=0.9,1.1,1.3,1.5' \
    'model.kwargs.threshold_j=1.4,1.0' \
    'model.kwargs.threshold_in=1.0,1.4' \
    'optimizer.learning_rate_wout=0.05,0.01,0.005' \
    'optimizer.learning_rate_win=0.05,0.01,0.005' \
    'optimizer.learning_rate_j=0.01,0.005,0.001' \
    --multirun