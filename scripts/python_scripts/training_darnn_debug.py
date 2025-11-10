# run_debug.py — hydra-free debug runner
from typing import TYPE_CHECKING
import os
import time
import logging
from pprint import pformat
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset
from local_exp.trainers.registry import build_trainer
from local_exp.utils.evals import compute_overlaps
from local_exp.logging.logging_utils import Logger

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

if TYPE_CHECKING:
    from darnax.trainers.interface import Trainer


# ---------- DEFAULT CONFIG (edit these for your debug runs) ----------
DEFAULT_CFG = {
    "master_seed": 0,
    "epochs": 3,  # keep small for debugging
    "model": {
        "name": "fc-baseline",
        "kwargs": {
            "seed": 44,
            "dim_data": 784,
            "dim_hidden": 1024,
            "num_labels": 10,
            "strength_forth": 1.0,
            "strength_back": 0.5,
            "threshold_in": 1.0,
            "threshold_j": 1.0,
            "threshold_out": 0.0,
            "j_d": 0.5,
        },
    },
    "data": {
        "name": "general_mnist",
        "kwargs": {
            "batch_size": 64,
            "linear_projection": None,
            "num_images_per_class": None,
        },
    },
    "optimizer": {
        "learning_rate": 1e-3,
    },
    "trainer": {
        "name": "dynamical",
        "kwargs": {},
    },
    "wandb": {
        "enabled": True,
        "entity": "willinki-bocconi-university",
        "project": "darnax-new-benchmarks-v2",
        "run_name": "debug",
        "mode": "online",
        "tags": ["debug"],
        "dir": "./debug",
    },
}
# --------------------------------------------------------------------


def _to_namespace(obj):
    """Recursively convert dicts to SimpleNamespace for attribute-style access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_namespace(x) for x in obj)
    return obj


def main():
    # Build config namespace
    cfg = _to_namespace(DEFAULT_CFG)

    logger.info(f"Working dir: {os.getcwd()}")
    logger.info("Config (defaults at top of file):\n" + pformat(DEFAULT_CFG))

    master_key = jax.random.key(seed=cfg.master_seed)
    # wandb_logger = Logger(cfg)

    # Build model from registry
    state, orchestrator = build_model(
        cfg.model.name,
        **(
            cfg.model.kwargs.__dict__
            if isinstance(cfg.model.kwargs, SimpleNamespace)
            else cfg.model.kwargs
        ),
    )

    # Build dataset from registry
    ds = build_dataset(
        cfg.data.name,
        **(
            cfg.data.kwargs.__dict__
            if isinstance(cfg.data.kwargs, SimpleNamespace)
            else cfg.data.kwargs
        ),
    )
    master_key, data_key = jax.random.split(master_key)
    ds.build(data_key)

    # Build optimizer (Adam for now)
    optimizer = optax.adam(learning_rate=cfg.optimizer.learning_rate)
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

    # Build trainer
    trainer: "Trainer" = build_trainer(
        cfg.trainer.name,
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **(
            cfg.trainer.kwargs.__dict__
            if isinstance(cfg.trainer.kwargs, SimpleNamespace)
            else cfg.trainer.kwargs
        ),
    )

    #
    # TRAINING STARTS HERE
    #
    # Note: original loop used range(1, cfg.epochs); keeping behavior.
    for epoch in range(1, cfg.epochs):
        t0 = time.time()
        logger.info(f"\n=== Epoch {epoch}/{cfg.epochs} ===")

        # Training epoch
        logger.info("Training")
        for x_batch, y_batch in ds:
            master_key = trainer.train_step(x_batch, y_batch, master_key)

        # Evaluation epoch
        logger.info("Evaluating and computing overlaps")
        accs = []
        overlaps_batches = []
        for x_b, y_b in ds.iter_test():
            master_key, metrics = trainer.eval_step(x_b, y_b, master_key)
            accs.append(metrics["accuracy"])
            overlaps = compute_overlaps(
                trainer.orchestrator, trainer.state, x_b, y_b, master_key
            )
            overlaps_batches.append(overlaps)

        acc_eval = float(jnp.mean(jnp.array(accs))) if accs else float("nan")

        # Evaluation epoch on training set
        accs = []
        for x_b, y_b in ds:
            master_key, metrics = trainer.eval_step(x_b, y_b, master_key)
            accs.append(metrics["accuracy"])
        acc_train = float(jnp.mean(jnp.array(accs))) if accs else float("nan")

        logger.info(
            f"Train Accuracy = {acc_train:.3f} | Eval Accuracy = {acc_eval:.3f}"
            f" | epoch time: {time.time() - t0:.2f}s"
        )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
