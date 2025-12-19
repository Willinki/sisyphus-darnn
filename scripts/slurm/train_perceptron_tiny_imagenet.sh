#!/bin/bash -e
#SBATCH --job-name=train-mlp
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=train-mlp_%j.out
#SBATCH --error=train-mlp_%j.err

#PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
#mkdir -p "${PROJECT_ROOT}/logs"
#source "${PROJECT_ROOT}/scripts/slurm/constants.sh"
#
#module load anaconda3/2024.02
#source "$(conda info --base)/etc/profile.d/conda.sh"

#conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/python_scripts/training_mlp.py \
uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=312 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=726 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=1208 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=1208 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=1770 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=2412 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16

uv run python scripts/python_scripts/training_perceptron.py \
    data.name='tiny_imagenet' \
    'master_seed=44,55,66,77' \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.label_mode='ooe' \
    model.kwargs.input_dim=512 \
    model.kwargs.hidden_dim=3133 \
    model.kwargs.output_dim=200 \
    model.kwargs.lr=0.0001 \
    epochs=20 \
    'wandb.tags=[baseline,scalingH,emnist,non-hetero]' \
    'wandb.project=tiny_imagenet-scaling' \
    model.kwargs.use_bias=false \
    model.kwargs.loss_type=cross_entropy \
    data.kwargs.batch_size=16
