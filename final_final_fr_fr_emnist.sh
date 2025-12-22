python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=6000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.25

python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=5000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75

python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=4000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.0

python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=3000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75

python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=1000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=3.5

python scripts/python_scripts/main.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    model.kwargs.dim_hidden=2000 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.5