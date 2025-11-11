#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn_with_lr.py \
    experiment=mnist_dynamical_wback \
    model.kwargs.threshold_back=1.0 \
    optimizer.special.learning_rate=0.0000