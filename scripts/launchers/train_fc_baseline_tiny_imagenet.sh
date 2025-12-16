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
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=1155 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0007 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=3.5,2.7' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=1660 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0005 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=3.0,2.0' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=2216 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0005 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=3.0,2.0' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=2817 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0005 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=2.5,2.0' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=3458 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0005 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=2.5,2.0' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5

uv run python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_tinyimagenet_tuning \
    model.kwargs.dim_hidden=4134 \
    epochs=20 \
    torch_clf.epochs=20 \
    torch_clf.lr=0.0005 \
    "wandb.tags=[ours,tiny-imagenet,scaling-h]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=512 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.90 \
    model.kwargs.threshold_in=1.6 \
    model.kwargs.threshold_j=1.4 \
    model.kwargs.threshold_out=7.0 \
    model.kwargs.strength_forth=5.00 \
    'model.kwargs.strength_back=2.5,2.0' \
    optimizer.weight_decay_j=0.01 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.02 \
    optimizer.learning_rate_win=0.3 \
    optimizer.learning_rate_wout=0.02 \
    trainer.fake_dynamics.k=0.5