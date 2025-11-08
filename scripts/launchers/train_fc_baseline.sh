#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=cifar10_features_s_dynamical \
    data.kwargs.x_transform="identity" \
    'model.kwags.dim_hidden=100,200,400,800,1600,3200,6400' \
    'model.kwargs.strength_forth=5.0,4.0,3.0' \
    'model.kwargs.strength_back=0.9,1.1,0.5' \
    'model.kwargs.threshold_in=1.2,1.0,0.8' \
    'model.kwargs.threshold_j=1.2,1.0,0.8' \
    'model.kwargs.threshold_out=1.2,1.0,0.8' \
    'model.kwargs.j_d=0.3,0.5,0.7' \
    --multirun


    