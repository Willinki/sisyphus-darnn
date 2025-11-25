#!/bin/bash -e

uv run python scripts/python_scripts/main.py \
   model.kwargs.dim_hidden=1000 \
   epochs=20 \
   --multirun \
   'wandb.tags=[ours,mnist,tuning]' \
   data.kwargs.x_transform='identity' \
   data.kwargs.linear_projection=null \
   model.name=fc-baseline-sparse-fully \
   model.kwargs.dim_data=784 \
   model.kwargs.sparsity=0.99 \
   model.kwargs.sparsity_win=0.9 \
   'model.kwargs.threshold_j=1.4,1.8' \
   'model.kwargs.j_d=0.9,0.7' \
   'model.kwargs.threshold_in=1.6,2.0' \
   'model.kwargs.strength_forth=5.0,4.0' \
   'optimizer.learning_rate_win=0.3,0.05' \
   optimizer.weight_decay_j=0.01 \
   'optimizer.learning_rate_j=0.02,0.1' \
   trainer.fake_dynamics.k=0.5
