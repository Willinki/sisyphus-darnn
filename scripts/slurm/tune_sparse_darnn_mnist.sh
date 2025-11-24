#!/bin/bash
#SBATCH --job-name=mnist-sparse-grid
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=mnist-sparse_%A_%a.out
#SBATCH --error=mnist-sparse_%A_%a.err
#SBATCH --array=0-1727%4 this_script.sh

# ----------------------------------------------
# Parameter grids (edit values as needed)
# ----------------------------------------------
sf_vals=(5.0 4.0 3.5 3.0)         # model.kwargs.strength_forth
sb_vals=(0.9 1.1 1.3 1.5)         # model.kwargs.strength_back
thj_vals=(1.4 1.0)                # model.kwargs.threshold_j
thin_vals=(1.0 1.4)               # model.kwargs.threshold_in
lr_wout_vals=(0.05 0.01 0.005)    # optimizer.learning_rate_wout
lr_win_vals=(0.05 0.01 0.005)     # optimizer.learning_rate_win
lrj_vals=(0.01 0.005 0.001)       # optimizer.learning_rate_j

len_sf=${#sf_vals[@]}
len_sb=${#sb_vals[@]}
len_thj=${#thj_vals[@]}
len_thin=${#thin_vals[@]}
len_lr_wout=${#lr_wout_vals[@]}
len_lr_win=${#lr_win_vals[@]}
len_lrj=${#lrj_vals[@]}

total=$((len_sf * len_sb * len_thj * len_thin * len_lr_wout * len_lr_win * len_lrj))

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "This script is intended to be submitted as an array job."
  echo "Submit with: sbatch --array=0-$((total-1))%50 $0"
  echo "(total combinations = $total)"
  exit 1
fi

idx=$SLURM_ARRAY_TASK_ID
if (( idx < 0 || idx >= total )); then
  echo "SLURM_ARRAY_TASK_ID ($idx) out of range (0..$((total-1)))"
  exit 2
fi

# ----------------------------------------------
# Map linear index -> grid indices (mixed radix)
# Order: sf, sb, thj, thin, lr_wout, lr_win, lrj
# ----------------------------------------------
stride=$((len_sb * len_thj * len_thin * len_lr_wout * len_lr_win * len_lrj))
i_sf=$(( idx / stride % len_sf ))

stride=$((len_thj * len_thin * len_lr_wout * len_lr_win * len_lrj))
i_sb=$(( idx / stride % len_sb ))

stride=$((len_thin * len_lr_wout * len_lr_win * len_lrj))
i_thj=$(( idx / stride % len_thj ))

stride=$((len_lr_wout * len_lr_win * len_lrj))
i_thin=$(( idx / stride % len_thin ))

stride=$((len_lr_win * len_lrj))
i_lr_wout=$(( idx / stride % len_lr_wout ))

stride=$((len_lrj))
i_lr_win=$(( idx / stride % len_lr_win ))

i_lrj=$(( idx % len_lrj ))

sf=${sf_vals[i_sf]}
sb=${sb_vals[i_sb]}
thj=${thj_vals[i_thj]}
thin=${thin_vals[i_thin]}
lr_wout=${lr_wout_vals[i_lr_wout]}
lr_win=${lr_win_vals[i_lr_win]}
lr_j=${lrj_vals[i_lrj]}

echo "Array task $SLURM_ARRAY_TASK_ID/$((total-1))"
echo "  strength_forth=$sf strength_back=$sb threshold_j=$thj threshold_in=$thin"
echo "  lr_wout=$lr_wout lr_win=$lr_win lr_j=$lr_j"

# ----------------------------------------------
# Run
# ----------------------------------------------
uv run python3.11 scripts/python_scripts/main.py \
  --config-name=ours_sparse_mnist \
  model.kwargs.dim_hidden=1000 \
  epochs=7 \
  model.kwargs.j_d=0.7 \
  model.kwargs.sparsity=0.99 \
  model.kwargs.sparsity_win=0.90 \
  model.kwargs.strength_forth="${sf}" \
  model.kwargs.strength_back="${sb}" \
  model.kwargs.threshold_j="${thj}" \
  model.kwargs.threshold_in="${thin}" \
  optimizer.learning_rate_wout="${lr_wout}" \
  optimizer.learning_rate_win="${lr_win}" \
  optimizer.learning_rate_j="${lr_j}"
