# search_optuna_asha.py  (Ray-free single run)
from typing import Dict, Any
import os, time
from datetime import datetime
from copy import deepcopy
import uuid
import logging

import wandb
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset
from local_exp.trainers.registry import build_trainer
from config_dict import BASE_CONFIG

from metrics_debug import DEBUG_METRICS
from debug_runtime import (
    init_debug_buckets,
    update_debug_buckets,
    aggregate_debug_buckets,
    flatten_for_logging,
)

logger = logging.getLogger(__name__)


def train_once(cfg: Dict[str, Any]) -> None:
    cfg = deepcopy(cfg)
    key = jax.random.key(cfg.get("master_seed", 0))
    wb = cfg["wandb"]
    if wb.get("enabled", True):
        wandb.init(
            entity=wb["entity"],
            project=wb["project"],
            name=wb["run_name"],
            mode=wb["mode"],
            dir=wb["dir"],
            config=deepcopy(cfg),
            tags=list(wb.get("tags", [])),
            save_code=wb.get("save_code", True),
        )

    state, orchestrator = build_model(cfg["model"]["name"], **cfg["model"]["kwargs"])
    ds = build_dataset(cfg["data"]["name"], **cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    optimizer = optax.adam(cfg["optimizer"]["learning_rate"])
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

    trainer = build_trainer(
        cfg["trainer"]["name"],
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **cfg["trainer"]["kwargs"],
    )

    for epoch in range(0, int(cfg["epochs"]) + 1):
        t0 = time.time()

        # ---- Train ----
        if epoch != 0:
            for xb, yb in ds:
                key = trainer.train_step(xb, yb, key)

        # ---- Eval (test) + per-batch debug ----
        accs_eval = []
        buckets = init_debug_buckets(DEBUG_METRICS)
        for b_index, (xb, yb) in enumerate(ds.iter_test()):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_eval.append(metrics["accuracy"])
            update_debug_buckets(
                buckets=buckets,
                debug_metrics=DEBUG_METRICS,
                batch_id=b_index,
                x=xb,
                y=yb,
                orchestrator=trainer.orchestrator,
                state=trainer.state,
            )

        acc_eval = float(jnp.mean(jnp.array(accs_eval))) if accs_eval else float("nan")
        aggregated_debug = aggregate_debug_buckets(buckets, DEBUG_METRICS)
        debug_log = flatten_for_logging(prefix="debug/", aggregated=aggregated_debug)

        # ---- Eval (train split) for train accuracy (unchanged) ----
        accs_train = []
        for b_index, (xb, yb) in enumerate(ds):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
        acc_train = (
            float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")
        )

        if wb.get("enabled", True):
            log_content = debug_log
            if epoch != 0:
                log_content |= {
                    "accuracy/train": acc_train,
                    "accuracy/eval": acc_eval,
                    "epoch_time_s": time.time() - t0,
                }
            wandb.log(
                log_content,
                step=epoch,
                commit=True,
            )

        print(
            f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f} | time={time.time()-t0:.2f}s"
        )

    if wb.get("enabled", True):
        wandb.finish()


if __name__ == "__main__":
    train_once(BASE_CONFIG)
