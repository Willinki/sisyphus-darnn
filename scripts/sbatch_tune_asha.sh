#!/bin/bash

#SBATCH --job-name=tune_asha
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tune_asha_%j.out
#SBATCH --error=logs/tune_asha_%j.err

# Load any necessary modules (adjust based on your cluster setup)
# module load python/3.11
# module load cuda/12.0

# Create logs directory if it doesn't exist
mkdir -p logs

uv run python scripts/python_scripts/tune_asha.py \
    --n-trials=100 \
    --n-jobs=10 \
    --study-name=emnist_bsize_optuna \
    --storage=sqlite:///optuna_mnist_sparse.db

echo "Tuning job completed at $(date)"
