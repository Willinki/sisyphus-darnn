#!/bin/bash
uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=cifar10_features_s_perceptron \
    cluster=gpu \
    model.kwargs.optim=sgd \
    'model.kwargs.proj_dim=128,256,512,1024,2048,4096' \
    'model.kwargs.margin=1.0,0.8,1.2' \
    'model.kwargs.lr=0.001,0.0002,0.005' \
    'data.kwargs.x_transform=identity,sign' \
    --multirun