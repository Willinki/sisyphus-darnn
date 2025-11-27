#!/bin/bash

uv run python scripts/python_scripts/main.py \
    --config-name=ours_sparse_mnist_tuning \
    --multirun \
    model.kwargs.dim_hidden=1000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,mnist,tuning]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.9 \
    model.kwargs.threshold_in=1.7 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=3.8 \
    'model.kwargs.strength_forth=4.7,5.0' \
    model.kwargs.strength_back=1.7 \
    'optimizer.weight_decay_j=0.0004,0.0001' \
    'optimizer.weight_decay_win=0.001,0.01' \
    optimizer.weight_decay_wout=0.02 \
    'optimizer.learning_rate_j=0.05,0.1,0.25' \
    'optimizer.learning_rate_win=0.15,0.25' \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5
