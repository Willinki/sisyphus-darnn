import os
from typing import Any

BASE_CONFIG: dict[str, Any] = {
    "epochs": 100,
    "optimizer": {
        "learning_rate_wout": 0.005,
        "learning_rate_j": 0.1,
        "learning_rate_win": 0.14,
        "weight_decay_win": 0.00,
        "weight_decay_wout": 0.00,
        "weight_decay_j": 0.01,
    },
    "data": {
        "name": "general_mnist",
        "kwargs": {
            "batch_size": 16,
            "num_images_per_class": None,
            "x_transform": "identity",
            "linear_projection": None,
            "label_mode": "pm1",
        },
    },
    "trainer": {
        "name": "dynamical_v2",
        "kwargs": {
            "warmup_n_iter": 1,
            "train_clamped_n_iter": 8,
            "train_free_n_iter": 8,
            "eval_n_iter": 8,
        },
    },
    "model": {
        "name": "fc-baseline-rescale",
        "kwargs": {
            "seed": 99,
            "dim_data": 784,
            "dim_hidden": 128,
            "num_labels": 10,
            # "sparsity": 0.5,
            "strength_forth": 4.1,
            "strength_back": 0.93,
            "threshold_in": 0.85,
            "threshold_j": 1.01,
            "threshold_out": 1.5,
            "j_d": 0.5,
        },
    },
    "wandb": {
        "enabled": True,
        "entity": "willinki-bocconi-university",
        "project": "darnax-new-benchmarks-v2",
        "run_name": f"emnist-sparse",
        "mode": "online",
        "tags": ["fc-baseline", "emnist", "dynamical-sparse-feedback"],
        "dir": os.getenv("WANDB_DIR", "./wandb_logs"),
        "save_code": True,
        "reinit": False,
    },
    "master_seed": 44,
}
