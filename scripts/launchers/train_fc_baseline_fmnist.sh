#!/bin/bash
#SBATCH --job-name=train-ours
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=train-mlp_%j.out
#SBATCH --error=train-mlp_%j.err

# final good configuration. last tests
uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    'master_seed=55,66,18' \
    model.kwargs.dim_hidden=916 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,2.1,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=1858 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,2.1,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    'master_seed=55,66,18' \
    model.kwargs.dim_hidden=2815 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,0.9,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=3783 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,0.9,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=4758 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,0.9,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --config-name=ours_sparse_fmnist_tuning \
    model.kwargs.dim_hidden=5737 \
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
    model.kwargs.j_d=0.986 \
    model.kwargs.threshold_in=1.294 \
    model.kwargs.threshold_j=1.74 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=4.91 \
    'model.kwargs.strength_back=1.84,0.9,1.5,1.2' \
    optimizer.weight_decay_j=0.0001 \
    optimizer.weight_decay_win=0.00002 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.118 \
    optimizer.learning_rate_win=0.17 \
    optimizer.learning_rate_wout=0.1 \
    trainer.fake_dynamics.k=0.5
