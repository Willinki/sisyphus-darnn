python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=640 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    wandb.project='emnist_scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=126 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=233 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=336 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=438 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='sign' \
    data.kwargs.linear_projection=100 \
    model.kwargs.input_dim=100 \
    model.kwargs.hidden_dim=540 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    wandb.project='emnist_scaling' \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16