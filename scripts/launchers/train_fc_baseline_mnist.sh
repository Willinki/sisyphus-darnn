#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    'model.kwargs.dim_hidden=128,256,512,1024,2048' \
    'model.kwargs.strength_forth=5.0,4.0,3.0' \
    'model.kwargs.strength_back=0.9,1.1,0.5' \
    'model.kwargs.j_d=0.3,0.5,0.7' \
    --multirun