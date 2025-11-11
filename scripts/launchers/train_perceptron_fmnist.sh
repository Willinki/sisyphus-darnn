#!/bin/bash
uv run python3.11 scripts/python_scripts/training_mlp.py \
   experiment=fashionmnist_perceptron \
   cluster=gpu \
   data.kwargs.x_transform=identity \
   'model.kwargs.proj_dim=128,256,512,1024,2048' \
   'model.kwargs.margin=1.0,0.8,1.2' \
   'model.kwargs.lr=0.001,0.0002,0.005' \
   --multirun
