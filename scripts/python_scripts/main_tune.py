import copy
import logging
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import optuna
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_exp.datasets.registry import build_dataset
from local_exp.models.registry import build_model
from local_exp.scratch.normalizer import decay
from local_exp.trainers.registry import build_trainer
from local_exp.utils.learning_rate import make_lr_map_v2

JaxArray = jax.Array


#
# DEBUG METRIC 3: weights
# logged as a histogram
def get_weights(batch_id, x, y, orchestrator, state):
    if batch_id != 0:
        return None

    return {
        "J": wandb.Histogram(orchestrator.lmap[1][1].J),
        "W_in": wandb.Histogram(orchestrator.lmap[1][0].W),
        "W_out": wandb.Histogram(orchestrator.lmap[2][1].W),
    }


def pass_weights(values):
    values = [x for x in values if x is not None]
    assert len(values) == 1
    return values[0]


DEBUG_METRICS = {
    "weights": (get_weights, pass_weights),
}


def init_debug_buckets(debug_metrics: Dict[str, Tuple[callable, callable]]):
    return {name: [] for name in debug_metrics.keys()}


def update_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
    batch_id: int,
    x: jax.Array,
    y: jax.Array,
    orchestrator: object,
    state: object,
):
    for name, (per_batch, _) in debug_metrics.items():
        buckets[name].append(per_batch(batch_id, x, y, orchestrator, state))


def aggregate_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
):
    aggregated = {}
    for name, (_, aggregate) in debug_metrics.items():
        aggregated[name] = aggregate(buckets[name])
    return aggregated


def flatten_for_logging(
    prefix: str, aggregated: Dict[str, object]
) -> Dict[str, object]:
    out = {}
    for k, v in aggregated.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{prefix}{k}/{kk}"] = vv
        else:
            out[f"{prefix}{k}"] = v
    return out


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Optuna utilities
# -------------------------------------------------------------------------
def apply_optuna_suggestions(trial: optuna.Trial, cfg):
    """
    Mutate cfg in-place with Optuna suggestions for:
    - learning rates
    - weight decays
    - strengths / thresholds / j_d under model.kwargs
    """

    # ---- Optimizer learning rates (JAX part) ----
    # You can adjust ranges as you like.
    cfg.optimizer.learning_rate_win = trial.suggest_float(
        "optimizer.learning_rate_win", 0.005, 0.1, log=True
    )
    cfg.optimizer.learning_rate_wout = trial.suggest_float(
        "optimizer.learning_rate_wout", 0.005, 0.1, log=True
    )
    cfg.optimizer.learning_rate_j = trial.suggest_float(
        "optimizer.learning_rate_j", 0.005, 0.1, log=True
    )
    cfg.optimizer.weight_decay_j = trial.suggest_float(
        "optimizer.weight_decay_j", 0.00001, 0.001, log=True
    )
    cfg.optimizer.weight_decay_win = trial.suggest_float(
        "optimizer.weight_decay_win", 1e-8, 0.001, log=True
    )
    cfg.optimizer.weight_decay_wout = trial.suggest_float(
        "optimizer.weight_decay_wout", 1e-8, 0.001, log=True
    )

    # ---- Model strengths / thresholds / j_d (under model.kwargs) ----
    cfg.model.kwargs.strength_back = trial.suggest_float(
        "model.kwargs.strength_back", 0.9, 2.5, log=False
    )
    cfg.model.kwargs.strength_back = trial.suggest_float(
        "model.kwargs.strength_forth", 3.0, 5.0, log=False
    )

    # Thresholds, if present — adjust names / ranges to your actual config.
    cfg.model.kwargs.threshold = trial.suggest_float(
        "model.kwargs.threshold_j", 0.9, 2.0, log=False
    )
    cfg.model.kwargs.threshold_up = trial.suggest_float(
        "model.kwargs.threshold_in", 0.9, 2.0, log=False
    )
    cfg.model.kwargs.threshold_down = trial.suggest_float(
        "model.kwargs.threshold_out", 2.0, 4.0, log=False
    )

    cfg.model.kwargs.j_d = trial.suggest_float("model.kwargs.j_d", 0.7, 1.2, log=False)

    return cfg


def run_hyperopt(base_cfg):
    """
    Run Optuna with ASHA-style pruning (SuccessiveHalvingPruner in Optuna)
    over the subset of hyperparameters defined in apply_optuna_suggestions.
    """

    n_trials = 100

    # ASHA-style pruner in Optuna is SuccessiveHalvingPruner. :contentReference[oaicite:0]{index=0}
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=5,  # minimum number of epochs
        reduction_factor=2,
        min_early_stopping_rate=0,
    )

    sampler = optuna.samplers.TPESampler()

    def objective(trial: optuna.Trial):
        # fresh copy of cfg for each trial
        cfg = deepcopy(base_cfg)

        # apply suggestions (LRs, decays, strengths, thresholds, j_d)
        cfg = apply_optuna_suggestions(trial, cfg)
        print("Starting trial with params:", trial.params)
        # run training, report back best eval accuracy
        best_eval = run_training(cfg, trial=trial)
        return best_eval

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=("asha_optuna_study"),
    )
    study.optimize(objective, n_trials=n_trials)

    print("=== Optuna + ASHA finished ===")
    print("Best trial:", study.best_trial.number)
    print("Best value (eval accuracy):", study.best_value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


# -------------------------------------------------------------------------
# Core training (refactored out of hydra.main so Optuna can call it)
# -------------------------------------------------------------------------
def run_training(cfg, trial: optuna.Trial | None = None) -> float:
    """
    One complete training run.

    If `trial` is not None, we:
    - report eval accuracy to Optuna every epoch
    - allow ASHA pruning via trial.should_prune()

    Returns:
        best_eval_acc (float): best eval accuracy across epochs.
    """
    print("beginning...")
    cfg = deepcopy(cfg)
    key = jax.random.key(cfg.get("master_seed", 0))
    wb = cfg["wandb"]
    if wb.get("enabled", True):
        cfg_for_wandb = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            entity=wb["entity"],
            project=wb["project"],
            name="optuna-" + wb["run_name"],
            mode=wb["mode"],
            dir=wb["dir"],
            config=cfg_for_wandb,
            tags=list(wb.get("tags", [])),
            save_code=wb.get("save_code", True),
        )

    print("initialized wandb" if wb.get("enabled", True) else "wandb disabled")

    state, orchestrator = build_model(cfg["model"]["name"], **cfg["model"]["kwargs"])
    if cfg.trainer.gating.enabled:
        # set wout equal to rescaled transpose of wback
        # simulates wout warmup
        wback = orchestrator.lmap[1][2].W  # (C, H)
        wout = orchestrator.lmap[2][1].W  # (H, C)
        C, H = wback.shape[0], wback.shape[1]
        assert wout.shape[0] == H and wout.shape[1] == C
        scale_ratio = (C / H) ** 0.5 / cfg.model.kwargs.strength_back
        rescaled_transpose_wback = copy.deepcopy(wback).T * scale_ratio
        orchestrator = eqx.tree_at(
            lambda x: x.lmap[2][1].W, orchestrator, rescaled_transpose_wback
        )
    ds = build_dataset(cfg["data"]["name"], **cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    lr_map = make_lr_map_v2(
        orchestrator,
        overrides={(1, 0): "w_in", (1, 1): "j", (2, 1): "w_out", (1, 2): "w_back"},
    )
    lr_win = cfg["optimizer"]["learning_rate_win"]
    lr_wout = cfg["optimizer"]["learning_rate_wout"]
    lr_j = cfg["optimizer"]["learning_rate_j"]
    lr_wback = (
        0.0
        if not cfg.model.kwargs.learnable_wback
        else cfg["optimizer"]["learning_rate_wback"]
    )
    if cfg.model.name in ["fc-baseline-sparse", "fc-baseline-sparse-fully"]:
        print("Rescaling learning rates to account for sparsity...")
        lr_j /= jnp.sqrt(1 - cfg.model.kwargs.sparsity)
    if cfg.model.name in ["fc-baseline-sparse-fully"]:
        lr_win /= jnp.sqrt(1 - cfg.model.kwargs.sparsity_win)
    optimizer = optax.multi_transform(
        {
            "default": optax.sgd(learning_rate=0.0),
            "w_in": optax.sgd(learning_rate=lr_win),
            "w_out": optax.sgd(learning_rate=lr_wout),
            "j": optax.sgd(learning_rate=lr_j),
            "w_back": optax.sgd(learning_rate=lr_wback),
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

    print("initialized trainer")

    best_eval_acc = float("-inf")

    # ---- saving norms ----
    for epoch in range(0, int(cfg["epochs"]) + 1):
        t0 = time.time()

        # ---- Train ----
        if epoch != 0:
            count = 0
            for xb, yb in ds:
                use_gating = cfg.trainer.gating.enabled and (
                    epoch > cfg.trainer.gating.warmup_epochs
                )
                key = trainer.train_step(
                    xb,
                    yb,
                    key,
                    use_gating=use_gating,
                    gating_shift=cfg.trainer.gating.shift,
                    fake_dynamics=cfg.trainer.fake_dynamics.enabled,
                    fake_dynamics_k=cfg.trainer.fake_dynamics.k,
                    fake_dynamics_vanilla=cfg.trainer.fake_dynamics.vanilla,
                    double_dynamics=cfg.trainer.double_dynamics,
                )
                trainer.orchestrator = decay(trainer.orchestrator, cfg)

                count += 1

        # ---- Eval (test) + per-batch debug ----
        accs_eval = []
        for b_index, (xb, yb) in enumerate(ds.iter_test()):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_eval.append(metrics["accuracy"])

        acc_eval = float(jnp.mean(jnp.array(accs_eval))) if accs_eval else float("nan")

        # ---- Eval (train split) for train accuracy (unchanged) ----
        accs_train = []
        buckets = init_debug_buckets(DEBUG_METRICS)
        for b_index, (xb, yb) in enumerate(ds):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
            update_debug_buckets(
                buckets=buckets,
                debug_metrics=DEBUG_METRICS,
                batch_id=b_index,
                x=xb,
                y=yb,
                orchestrator=trainer.orchestrator,
                state=trainer.state,
            )
        acc_train = (
            float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")
        )
        aggregated_debug = aggregate_debug_buckets(buckets, DEBUG_METRICS)
        debug_log = flatten_for_logging(prefix="debug/", aggregated=aggregated_debug)

        # Track best eval metric
        if not np.isnan(acc_eval):
            best_eval_acc = max(best_eval_acc, acc_eval)

        # Report to Optuna (for ASHA pruning)
        if trial is not None:
            trial.report(acc_eval, step=epoch)
            if trial.should_prune():
                if wb.get("enabled", True):
                    wandb.finish()
                raise optuna.TrialPruned()

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
            f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f} | time={time.time() - t0:.2f}s"
        )

    # ---- Optional PyTorch linear classifier on learned representations ----
    # (Skip this during hyperopt to keep trials cheap)
    if trial is None and "torch_clf" in cfg and cfg.torch_clf.get("enabled", False):
        torch_clf_cfg = cfg.torch_clf
        print("Starting PyTorch linear classifier training...")
        # Seed
        if "master_seed" in cfg:
            torch.manual_seed(int(cfg.master_seed))

        # Feature extraction: TRAIN split
        train_reps = []
        train_labels = []
        for xb, yb in ds:  # train split
            key, _ = trainer.eval_step(xb, yb, key)
            reps = trainer.state.representations
            train_reps.append(copy.deepcopy(np.array(reps)))
            yb_np = np.array(copy.deepcopy(yb))
            yb_np = np.argmax(yb_np, axis=-1)
            train_labels.append(copy.deepcopy(yb_np))

        # Feature extraction: TEST/EVAL split
        test_reps = []
        test_labels = []
        for xb, yb in ds.iter_test():
            key, _ = trainer.eval_step(xb, yb, key)
            reps = trainer.state.representations
            test_reps.append(copy.deepcopy(np.array(reps)))
            yb_np = np.array(copy.deepcopy(yb))
            yb_np = np.argmax(yb_np, axis=-1)
            test_labels.append(copy.deepcopy(yb_np))

        features_train = torch.from_numpy(np.concatenate(train_reps, axis=0)).float()
        labels_train = torch.from_numpy(np.concatenate(train_labels, axis=0)).long()
        features_test = torch.from_numpy(np.concatenate(test_reps, axis=0)).float()
        labels_test = torch.from_numpy(np.concatenate(test_labels, axis=0)).long()

        input_dim = int(features_train.shape[1])
        num_classes = int(cfg.model.kwargs.num_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(
            input_dim, num_classes, bias=cfg.torch_clf.get("use_bias", False)
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        opt = torch_clf_cfg.get("optimizer", "adam").lower()
        opt_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[opt]
        optimizer = opt_class(
            model.parameters(),
            lr=float(torch_clf_cfg.lr),
            weight_decay=float(torch_clf_cfg.weight_decay),
        )

        batch_size = int(torch_clf_cfg.batch_size)
        train_loader = DataLoader(
            TensorDataset(features_train, labels_train),
            batch_size=batch_size,
            shuffle=True,
        )

        prefix = str(torch_clf_cfg.get("log_prefix", "torch_clf"))
        epochs_clf = int(torch_clf_cfg.epochs)
        base_step = int(cfg["epochs"]) + 1  # start after last JAX epoch index

        for e in range(epochs_clf):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for xb_t, yb_t in train_loader:
                xb_t = xb_t.to(device)
                yb_t = yb_t.to(device)
                optimizer.zero_grad()
                logits = model(xb_t)
                loss = criterion(logits, yb_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * yb_t.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb_t).sum().item()
                total += yb_t.size(0)
            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            model.eval()
            with torch.no_grad():
                logits_eval = model(features_test.to(device))
                eval_loss = criterion(logits_eval, labels_test.to(device)).item()
                pred_eval = logits_eval.argmax(dim=1)
                eval_acc = (pred_eval == labels_test.to(device)).float().mean().item()

            if cfg.wandb.get("enabled", True):
                wandb.log(
                    {
                        f"{prefix}/train_loss": train_loss,
                        f"{prefix}/train_acc": train_acc,
                        f"{prefix}/eval_loss": eval_loss,
                        f"{prefix}/eval_acc": eval_acc,
                        f"{prefix}/epoch": e,
                    },
                    step=base_step + e,
                    commit=True,
                )

            print(
                f"[Torch Clf] Epoch {e:03d} | train_acc={train_acc:.4f} | eval_acc={eval_acc:.4f} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}"
            )

        print("PyTorch linear classifier training complete.")

    if wb.get("enabled", True):
        wandb.finish()

    return best_eval_acc


# -------------------------------------------------------------------------
# Hydra entrypoint: normal run vs hyperopt run
# -------------------------------------------------------------------------
@hydra.main(
    version_base=None, config_path="../../configs", config_name="ours_entangled_mnist"
)
def train_once(cfg) -> None:
    """
    If cfg.hyperopt.enabled is True, run Optuna+ASHA sweep.
    Otherwise, just run a single training as before.
    """
    run_hyperopt(cfg)


if __name__ == "__main__":
    train_once()
