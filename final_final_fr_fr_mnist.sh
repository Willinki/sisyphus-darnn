python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_mnist_tuning \
    model.kwargs.dim_hidden='5000,6000' \
    master_seed='44,99,66,44,80' \
    epochs=20 \
    wandb.project="mnist-scaling" \
    torch_clf.epochs=20 \
    data.kwargs.x_transform=identity \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.95 \
    model.kwargs.threshold_in=1.78 \
    model.kwargs.threshold_j=1.78 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    model.kwargs.strength_back=0.7 \
    optimizer.weight_decay_j=0.00006 \
    optimizer.weight_decay_win=0.01 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.058 \
    optimizer.learning_rate_win=0.159 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_mnist_tuning \
    model.kwargs.dim_hidden=4000 \
    master_seed='44,99,66,44,80' \
    epochs=20 \
    wandb.project="mnist-scaling" \
    torch_clf.epochs=20 \
    data.kwargs.x_transform=identity \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.95 \
    model.kwargs.threshold_in=1.78 \
    model.kwargs.threshold_j=1.78 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    model.kwargs.strength_back=0.8 \
    optimizer.weight_decay_j=0.00006 \
    optimizer.weight_decay_win=0.01 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.058 \
    optimizer.learning_rate_win=0.159 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

python scripts/python_scripts/main.py \
    --multirun \
    --config-name=ours_sparse_mnist_tuning \
    model.kwargs.dim_hidden=3000 \
    master_seed='44,99,66,44,80' \
    epochs=20 \
    wandb.project="mnist-scaling" \
    torch_clf.epochs=20 \
    data.kwargs.x_transform=identity \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d=0.95 \
    model.kwargs.threshold_in=1.78 \
    model.kwargs.threshold_j=1.78 \
    model.kwargs.threshold_out=7 \
    model.kwargs.strength_forth=5.0 \
    model.kwargs.strength_back=1.1 \
    optimizer.weight_decay_j=0.00006 \
    optimizer.weight_decay_win=0.01 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j=0.058 \
    optimizer.learning_rate_win=0.159 \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5