python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    'model.kwargs.hidden_dim=300,400' \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &


python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    'model.kwargs.hidden_dim=500,600' \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=700 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=800 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=900 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16 &

python scripts/python_scripts/training_mlp.py \
    --multirun \
    --config-name=binary_mlp_tinyimagenet \
    'master_seed=33,44,55,66,77' \
    model.name="binary-3layer-mlp" \
    data.name='tiny_imagenet' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=1000 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,tiny_imagenet,non-hetero]' \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16