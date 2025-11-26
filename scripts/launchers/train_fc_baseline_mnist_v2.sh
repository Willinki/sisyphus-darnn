#!/bin/bash

threshold_js=(1.5)
threshold_ins=(1.5)
j_ds=(0.98)
strength_forths=(3.0)
lr_wins=(0.02)
lr_js=(0.03)

for tj in "${threshold_js[@]}"; do
for jd in "${j_ds[@]}"; do
for tin in "${threshold_ins[@]}"; do
for sf in "${strength_forths[@]}"; do
for lrw in "${lr_wins[@]}"; do
for lrj in "${lr_js[@]}"; do

  echo "Running tj=$tj jd=$jd tin=$tin sf=$sf lrw=$lrw lrj=$lrj"

  uv run python scripts/python_scripts/main.py \
    model.kwargs.dim_hidden=1000 \
    epochs=20 \
    torch_clf.epochs=20 \
    "wandb.tags=[ours,mnist,tuning]" \
    data.kwargs.x_transform='identity' \
    data.kwargs.linear_projection=null \
    data.kwargs.batch_size=16 \
    model.name=fc-baseline-sparse-fully \
    model.kwargs.dim_data=784 \
    model.kwargs.sparsity=0.99 \
    model.kwargs.sparsity_win=0.9 \
    model.kwargs.j_d="$jd" \
    model.kwargs.threshold_in="$tin" \
    model.kwargs.threshold_j="$tj" \
    model.kwargs.threshold_out=3.8 \
    model.kwargs.strength_forth="$sf" \
    model.kwargs.strength_back=0.98 \
    optimizer.weight_decay_j=0.0004 \
    optimizer.weight_decay_win=0.0 \
    optimizer.weight_decay_wout=0.02 \
    optimizer.learning_rate_j="$lrj" \
    optimizer.learning_rate_win="$lrw" \
    optimizer.learning_rate_wout=0.17 \
    trainer.fake_dynamics.k=0.5

done
done
done
done
done
done
