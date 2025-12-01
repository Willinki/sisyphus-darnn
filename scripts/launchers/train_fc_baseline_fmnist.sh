#!/bin/bash

# final good configuration. last tests
uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=1000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=2000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=3000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=4000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=5000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=6000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,fmnist,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.21 \
    model.kwargs.threshold_j=1.7 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    'model.kwargs.strength_back=1.62,1.3,1.9,1.0' \
    optimizer.weight_decay_j=0.00001 \
    optimizer.weight_decay_win=0.0001 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.075 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5
