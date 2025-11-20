#!/bin/bash
#SBATCH --job-name=darnn_tuning
#SBATCH --partition=long_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --array=0-4%4

# Parameter grid (index matches SLURM_ARRAY_TASK_ID)
sparsities=(0.0 0.1 0.5 0.8 0.9)

# If SLURM_ARRAY_TASK_ID is not set (run locally), default to 0
idx=${SLURM_ARRAY_TASK_ID:-0}

dim_hidden=${dims[$idx]}
sparsity=${sparsities[$idx]}

echo "Task index: ${idx} -> dim_hidden=256, sparsity=${sparsity}"

uv run python3.11 scripts/parameter_tuning/asha_scheduler.py --dim_hidden 256 --sparsity "${sparsity}"