import os
from typing import Any

BASE_CONFIG: dict[str, Any] = {
    "epochs": 100,
    "optimizer": {
        "learning_rate_wout": 0.1,
        "learning_rate_j": 0.005,
        "learning_rate_win": 0.085,
        "weight_decay_win": 0.0,
        "weight_decay_wout": 0.005,
        "weight_decay_j": 0.005,
    },
    "data": {
        "name": "general_mnist",
        "kwargs": {
            "batch_size": 16,
            "num_images_per_class": None,
            "x_transform": "sign",
            "linear_projection": 100,
            "label_mode": "c-rescale",
        },
    },
    "trainer": {
        "name": "dynamical",
        "kwargs": {
            "warmup_n_iter": 1,
            "train_clamped_n_iter": 4,
            "train_free_n_iter": 5,
            "eval_n_iter": 5,
        },
    },
    "model": {
        "name": "fc-baseline",
        "kwargs": {
            "seed": 99,
            "dim_data": 100,
            "dim_hidden": 100,
            "num_labels": 10,
            "strength_forth": 5.0,
            "strength_back": 0.9,
            "threshold_in": 1.4,
            "threshold_j": 1.4,
            "threshold_out": 3.0,
            "j_d": 0.5,
        },
    },
    "wandb": {
        "enabled": True,
        "entity": "willinki-bocconi-university",
        "project": "darnax-new-benchmarks-v2",
        "run_name": f"emnist-size100",
        "mode": "online",
        "tags": ["fc-baseline", "emnist", "dynamical"],
        "dir": os.getenv("WANDB_DIR", "./wandb_logs"),
        "save_code": True,
        "reinit": False,
    },
    "master_seed": 44,
}
