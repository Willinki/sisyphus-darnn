#!/bin/bash

echo "Running tj=1.5 jd=0.98 tin=1.5 sf=3.0 lrw=0.02 lrj=0.03"

uv run python scripts/python_scripts/main_tune.py \
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
  model.kwargs.j_d=0.98 \
  model.kwargs.threshold_in=1.5 \
  model.kwargs.threshold_j=1.5 \
  model.kwargs.threshold_out=3.8 \
  model.kwargs.strength_forth=3.0 \
  model.kwargs.strength_back=0.98 \
  optimizer.weight_decay_j=0.0004 \
  optimizer.weight_decay_win=0.0 \
  optimizer.weight_decay_wout=0.02 \
  optimizer.learning_rate_j=0.03 \
  optimizer.learning_rate_win=0.02 \
  optimizer.learning_rate_wout=0.17 \
  trainer.fake_dynamics.k=0.5
