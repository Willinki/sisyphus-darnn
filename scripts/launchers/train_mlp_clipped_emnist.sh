#!/bin/bash

uv run python3.11 scripts/python_scripts/training_mlp.py \
    experiment=mnist_clipped_mlp \
    cluster=gpu \
    epochs=100 \
    model.kwargs.optim=sgd \