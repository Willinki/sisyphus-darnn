#!/bin/bash
#SBATCH --job-name=train-mlp
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --output=train-mlp_%A_%a.out
#SBATCH --error=train-mlp_%A_%a.err
#SBATCH --array=0-63%10

# parameter grids (edit values as needed)
wd_vals=(0.005 0.02)
lrj_vals=(0.001 0.01)
thj_vals=(1.0 1.4)
thin_vals=(1.0 1.4)
sf_vals=(3.0 5.0)
lr_win_vals=(0.05 0.1)  

len_wd=${#wd_vals[@]}
len_lr=${#lrj_vals[@]}
len_thj=${#thj_vals[@]}
len_thin=${#thin_vals[@]}
len_sf=${#sf_vals[@]}
len_lr_win=${#lr_win_vals[@]}

total=$((len_wd * len_lr * len_thj * len_thin * len_sf * len_lr_win))

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "This script is intended to be submitted as an array job."
    echo "Submit with: sbatch --array=0-$((total-1)) $0"
    exit 1
fi

idx=$SLURM_ARRAY_TASK_ID
if (( idx < 0 || idx >= total )); then
    echo "SLURM_ARRAY_TASK_ID ($idx) out of range (0..$((total-1)))"
    exit 2
fi

# map linear index to grid indices (mixed-radix)
stride=$((len_lr * len_thj * len_thin * len_sf * len_lr_win))
i_wd=$(( idx / stride % len_wd ))

stride=$((len_thj * len_thin * len_sf * len_lr_win))
i_lr=$(( idx / stride % len_lr ))

stride=$((len_thin * len_sf * len_lr_win))
i_thj=$(( idx / stride % len_thj ))

stride=$((len_sf * len_lr_win))
i_thin=$(( idx / stride % len_thin ))

stride=$((len_lr_win))
i_sf=$(( idx / stride % len_sf ))

i_lr_win=$(( idx % len_lr_win ))

wd=${wd_vals[i_wd]}
lr=${lrj_vals[i_lr]}
thj=${thj_vals[i_thj]}
thin=${thin_vals[i_thin]}
sf=${sf_vals[i_sf]}
lr_win=${lr_win_vals[i_lr_win]}

echo "Array task $SLURM_ARRAY_TASK_ID/$((total-1)): weight_decay_j=$wd learning_rate_j=$lr threshold_j=$thj threshold_in=$thin strength_forth=$sf learning_rate_win=$lr_win"

# run command (adjust 'uv run' if needed)
uv run python3.11 scripts/python_scripts/main.py \
        --multirun \
        model.kwargs.dim_hidden=320 \
        epochs=20 \
        'wandb.tags=[ours,sparsity,grid-search]' \
        model.name=fc-baseline-sparse \
        +model.kwargs.sparsity=0.9 \
        optimizer.weight_decay_j=${wd} \
        optimizer.learning_rate_j=${lr} \
        model.kwargs.threshold_j=${thj} \
        optimizer.learning_rate_win=${lr_win} \
        model.kwargs.threshold_in=${thin} \
        model.kwargs.strength_forth=${sf} \
        model.kwargs.j_d=0.9