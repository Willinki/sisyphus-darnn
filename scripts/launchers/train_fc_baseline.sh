#!/bin/bash

uv run python3.11 scripts/python_scripts/training_darnn.py \
    experiment=cifar10_features_s_dynamical \
    data.kwargs.x_transform="identity" \

    