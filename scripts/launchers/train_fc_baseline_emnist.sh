#!/bin/bash -e

python scripts/python_scripts/main.py \
    model.kwargs.dim_hidden=1000 \
    epochs=5 \
    'wandb.tags=[ours,emnist]' \
    model.name=fc-baseline-sparse-fully \
    +model.kwargs.sparsity=0.99 \
    +model.kwargs.sparsity_win=0.9 \
    optimizer.weight_decay_j=0.01 \
    optimizer.learning_rate_j=0.02 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.j_d=0.9 \
    optimizer.learning_rate_win=0.3 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.strength_forth=5.0 \
    trainer.fake_dynamics.k=0.5
