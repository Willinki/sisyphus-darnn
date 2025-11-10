#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    model.kwargs.dim_hidden=128 \
    epoch=100 \
    model.kwargs.strength_forth=4.0 \
    model.kwargs.strength_back=0.9 \
    model.kwargs.j_d=0.5 \
    'optimizer.learning_rate=0.001,0.0002,0.005' \
    'model.kwargs.threshold_in=0.8,1.0,1.2' \
    'model.kwargs.threshold_j=0.8,1.0,1.2' \
    'model.kwargs.threshold_out=0.8,1.0,1.2' \
    --multirun

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    model.kwargs.dim_hidden=256 \
    epoch=100 \
    model.kwargs.strength_forth=4.0 \
    model.kwargs.strength_back=0.9 \
    'model.kwargs.j_d=0.5,0.7' \
    'optimizer.learning_rate=0.001,0.0002,0.005' \
    'model.kwargs.threshold_in=0.8,1.0,1.2' \
    'model.kwargs.threshold_j=0.8,1.0,1.2' \
    'model.kwargs.threshold_out=0.8,1.0,1.2' \
    --multirun

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    model.kwargs.dim_hidden=512 \
    epoch=100 \
    model.kwargs.strength_forth=5.0 \
    model.kwargs.strength_back=0.9 \
    model.kwargs.j_d=0.5 \
    'optimizer.learning_rate=0.001,0.0002,0.005' \
    'model.kwargs.threshold_in=0.8,1.0,1.2' \
    'model.kwargs.threshold_j=0.8,1.0,1.2' \
    'model.kwargs.threshold_out=0.8,1.0,1.2' \
    --multirun

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    model.kwargs.dim_hidden=1024 \
    epoch=100 \
    model.kwargs.strength_forth=4.0 \
    model.kwargs.strength_back=0.9 \
    'model.kwargs.j_d=0.7,0.9' \
    'optimizer.learning_rate=0.001,0.0002,0.005' \
    'model.kwargs.threshold_in=0.8,1.0,1.2' \
    'model.kwargs.threshold_j=0.8,1.0,1.2' \
    'model.kwargs.threshold_out=0.8,1.0,1.2' \
    --multirun

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=mnist_dynamical \
    model.kwargs.dim_hidden=2048 \
    epoch=100 \
    model.kwargs.strength_forth=4.0 \
    model.kwargs.strength_back=1.1 \
    'model.kwargs.j_d=0.7,0.9' \
    'optimizer.learning_rate=0.001,0.0002,0.005' \
    'model.kwargs.threshold_in=0.8,1.0,1.2' \
    'model.kwargs.threshold_j=0.8,1.0,1.2' \
    'model.kwargs.threshold_out=0.8,1.0,1.2' \
    --multirun
