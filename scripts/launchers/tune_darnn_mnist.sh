#!/bin/bash
#SBATCH --job-name=darnn_tuning
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --ntasks=1

uv run python3.11 scripts/parameter_tuning/asha_scheduler.py --dim_hidden 128
uv run python3.11 scripts/parameter_tuning/asha_scheduler.py --dim_hidden 256
uv run python3.11 scripts/parameter_tuning/asha_scheduler.py --dim_hidden 512
uv run python3.11 scripts/parameter_tuning/asha_scheduler.py --dim_hidden 1024