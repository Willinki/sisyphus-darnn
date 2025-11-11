#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=fashionmnist_dynamical \
    'model.kwargs.dim_hidden=128,256,512,1024' \
    'model.kwargs.strength_forth=5.0,4.0,3.0' \
    'model.kwargs.strength_back=0.9,1.1,0.3' \
    'model.kwargs.j_d=0.5,0.7,0.9' \
    --multirun
