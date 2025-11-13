import os
from typing import Any

BASE_CONFIG: dict[str, Any] = {
    "epochs": 100,
    "optimizer": {"learning_rate": 0.002},
    "data": {
        "name": "general_mnist",
        "kwargs": {
            "batch_size": 64,
            "num_images_per_class": None,
            "x_transform": "identity",
            "linear_projection": None,
            "label_mode": "pm1",
        },
    },
    "trainer": {
        "name": "dynamical",
        "kwargs": {
            "warmup_n_iter": 1,
            "train_clamped_n_iter": 10,
            "train_free_n_iter": 10,
            "eval_n_iter": 20,
        },
    },
    "model": {
        "name": "fc-baseline-rescale",
        "kwargs": {
            "seed": 99,
            "dim_data": 784,
            "dim_hidden": 128,
            "num_labels": 10,
            "strength_forth": 5.0,
            "strength_back": 0.5,
            "threshold_in": 0.8,
            "threshold_j": 0.8,
            "threshold_out": 1.2,
            "j_d": 0.5,
        },
    },
    "wandb": {
        "enabled": True,
        "entity": "willinki-bocconi-university",
        "project": "darnax-new-benchmarks-v2",
        "run_name": f"you_exp",
        "mode": "online",
        "tags": ["fc-baseline", "mnist", "dynamical"],
        "dir": os.getenv("WANDB_DIR", "./wandb_logs"),
        "save_code": True,
        "reinit": False,
    },
    "master_seed": 44,
}
