from typing import Dict, Any, Callable, Optional
from copy import deepcopy
import time, uuid, argparse, logging
from datetime import datetime

import wandb
import jax, jax.numpy as jnp
import equinox as eqx
import optax

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset
from local_exp.trainers.registry import build_trainer
from local_exp.utils.learning_rate import make_lr_map_v2
from config_dict import BASE_CONFIG
from local_exp.scratch.normalizer import decay

from metrics_debug import DEBUG_METRICS
from debug_runtime import (
    init_debug_buckets,
    update_debug_buckets,
    aggregate_debug_buckets,
    flatten_for_logging,
)

logger = logging.getLogger(__name__)


# ---------- hyperparameter overrides ----------
def _apply_overrides(cfg: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge hyperparameters into cfg with NO defaults:
    - learning rates: lr_win, lr_wout, lr_j
    - weight decays: wd_win, wd_wout, wd_j
    - thresholds: threshold_in, threshold_j, threshold_out
    - "fields": strength_forth, strength_back
    Missing keys will raise KeyError.
    """
    cfg = deepcopy(cfg)

    # 1) optimizer learning rates
    cfg["optimizer"]["learning_rate_win"] = hp["lr_win"]
    cfg["optimizer"]["learning_rate_wout"] = hp["lr_wout"]
    cfg["optimizer"]["learning_rate_j"] = hp["lr_j"]

    # 2) optimizer weight decays
    cfg["optimizer"]["weight_decay_j"] = hp["wd_j"]

    # 2.1) jd
    cfg["model"]["kwargs"]["j_d"] = hp["j_d"]

    # 3) thresholds (live in model.kwargs in your BASE_CONFIG)
    cfg["model"]["kwargs"]["threshold_in"] = hp["threshold_in"]
    cfg["model"]["kwargs"]["threshold_j"] = hp["threshold_j"]
    cfg["model"]["kwargs"]["threshold_out"] = hp["threshold_out"]

    # 4) "fields" → strength_forth & strength_back
    cfg["model"]["kwargs"]["strength_forth"] = hp["strength_forth"]
    cfg["model"]["kwargs"]["strength_back"] = hp["strength_back"]

    return cfg


def train_once(
    cfg: Dict[str, Any], report_fn: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    cfg = deepcopy(cfg)
    key = jax.random.key(cfg["master_seed"])
    wb = cfg["wandb"]

    if wb["enabled"]:
        wandb.init(
            entity=wb["entity"],
            project=wb["project"],
            name=f"{wb['run_name']}-{uuid.uuid4().hex[:8]}",
            mode=wb["mode"],
            dir=wb["dir"],
            config=deepcopy(cfg),
            tags=list(wb["tags"]),
            save_code=wb["save_code"],
            reinit=wb.get("reinit", False),
            group=wb.get("group", None),
        )

    state, orchestrator = build_model(cfg["model"]["name"], **cfg["model"]["kwargs"])
    ds = build_dataset(cfg["data"]["name"], **cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    # optimizer with parameter groups
    lr_map = make_lr_map_v2(
        orchestrator, overrides={(1, 0): "w_in", (1, 1): "j", (2, 1): "w_out"}
    )
    optimizer = optax.multi_transform(
        {
            "default": optax.sgd(learning_rate=0.0),
            "w_in": optax.sgd(learning_rate=cfg["optimizer"]["learning_rate_win"]),
            "w_out": optax.sgd(learning_rate=cfg["optimizer"]["learning_rate_wout"]),
            "j": optax.sgd(learning_rate=cfg["optimizer"]["learning_rate_j"]),
        },
        lr_map,
    )
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

        # Train
        if epoch != 0:
            for xb, yb in ds:
                key = trainer.train_step(xb, yb, key)
                # IMPORTANT: use cfg so tuned weight decays apply
                trainer.orchestrator = decay(trainer.orchestrator, cfg)

        # Eval (test) + per-batch debug
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

        # Eval (train) for train accuracy
        accs_train = []
        for b_index, (xb, yb) in enumerate(ds):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
        acc_train = (
            float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")
        )

        if wb["enabled"]:
            log_content = debug_log
            if epoch != 0:
                log_content |= {
                    "accuracy_train": acc_train,
                    "accuracy_eval": acc_eval,
                    "epoch_time_s": time.time() - t0,
                }
            wandb.log(log_content, step=epoch, commit=True)

        # ASHA reporting (objective: validation accuracy)
        if report_fn is not None and epoch != 0:
            report_fn(
                {"accuracy_eval": acc_eval, "accuracy_train": acc_train, "epoch": epoch}
            )

        print(
            f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f} | time={time.time()-t0:.2f}s"
        )

    if wb["enabled"]:
        wandb.finish()


# ---------- Ray Tune entrypoints ----------
def tune_trainable(hp: Dict[str, Any]) -> None:
    cfg = _apply_overrides(BASE_CONFIG, hp)
    # Optional: tag runs as a group in W&B if you like
    cfg["wandb"]["group"] = cfg["wandb"].get("group", "asha-search")

    def report_fn(metrics: Dict[str, float]) -> None:
        tune.report(
            {
                "accuracy_eval": metrics["accuracy_eval"],
                "accuracy_train": metrics["accuracy_train"],
                "epoch": metrics["epoch"],
            }
        )

    train_once(cfg, report_fn=report_fn)


def run_asha_search() -> None:
    max_epochs = int(BASE_CONFIG["epochs"])
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="accuracy_eval",
        mode="max",
        max_t=max_epochs,
        grace_period=25,
        reduction_factor=3,
    )

    # Search space focused on your targets
    param_space = {
        # learning rates
        "lr_win": tune.loguniform(1e-3, 2e-1),
        "lr_wout": tune.loguniform(1e-3, 2e-1),
        "lr_j": tune.loguniform(1e-4, 1e-1),
        # weight decays (explicit grid incl. zero)
        "wd_j": tune.loguniform(1e-8, 1e-2),
        # thresholds (model kwargs)
        "threshold_in": tune.uniform(0.5, 1.5),
        "threshold_j": tune.uniform(0.5, 1.5),
        "threshold_out": tune.uniform(1.0, 4.0),
        # "fields" -> strengths
        "strength_forth": tune.uniform(1, 5e0),
        "strength_back": tune.uniform(5e-1, 1.5),
        # jd
        "j_d": tune.uniform(0.3, 0.9),
    }

    tuner = tune.Tuner(
        tune.with_resources(tune_trainable, resources={"cpu": 10, "gpu": 1}),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=ConcurrencyLimiter(
                OptunaSearch(metric="accuracy_eval", mode="max"), max_concurrent=1
            ),
            num_samples=100,  # adjust to your budget
        ),
        run_config=air.RunConfig(
            name=f"asha_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            verbose=0,
        ),
        param_space=param_space,
    )
    results = tuner.fit()
    best = results.get_best_result(metric="accuracy_eval", mode="max")
    print("\n===== Best result =====")
    print(f"accuracy_eval: {best.metrics['accuracy_eval']}")
    print(f"config: {best.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim_hidden",
        type=int,
        default=None,
        help="Override BASE_CONFIG['model']['kwargs']['dim_hidden']",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    if args.dim_hidden is not None:
        BASE_CONFIG["model"]["kwargs"]["dim_hidden"] = args.dim_hidden
    if args.sparsity is not None:
        BASE_CONFIG["model"]["kwargs"]["sparsity"] = args.sparsity

    run_asha_search()
