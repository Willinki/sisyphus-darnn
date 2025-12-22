python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=626 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=109 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=214 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=319 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=422 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    'master_seed=33,44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    model.kwargs.input_dim=784 \
    model.kwargs.hidden_dim=525 \
    'model.kwargs.lr=0.005' \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16