# search_optuna_asha.py  (Ray-free single run)
from collections import defaultdict
import copy
from typing import Dict, Any
import time
from copy import deepcopy
import logging

import hydra
import wandb
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset
from local_exp.trainers.registry import build_trainer
from local_exp.utils.learning_rate import make_lr_map_v2
from local_exp.scratch.normalizer import decay

from metrics_debug import DEBUG_METRICS
from debug_runtime import (
    init_debug_buckets,
    update_debug_buckets,
    aggregate_debug_buckets,
    flatten_for_logging,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="ours_entangled_mnist"
)
def train_once(cfg) -> None:
    print("beginning...")
    cfg = deepcopy(cfg)
    key = jax.random.key(cfg.get("master_seed", 0))
    wb = cfg["wandb"]
    if wb.get("enabled", True):
        cfg_for_wandb = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            entity=wb["entity"],
            project=wb["project"],
            name=wb["run_name"],
            mode=wb["mode"],
            dir=wb["dir"],
            config=cfg_for_wandb,
            tags=list(wb.get("tags", [])),
            save_code=wb.get("save_code", True),
        )

    print("initialized wandb")
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

    print("initialized trainer")

    # ---- saving norms ----
    for epoch in range(0, int(cfg["epochs"]) + 1):
        t0 = time.time()

        # ---- Train ----
        if epoch != 0:
            count = 0
            avg_logs = defaultdict(float)
            for xb, yb in ds:
                use_gating = cfg.trainer.gating.enabled and (
                    epoch > cfg.trainer.gating.warmup_epochs
                )
                key, logs = trainer.train_step(
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
                for k, v in logs.items():
                    avg_logs[k] += v
            for k in avg_logs:
                avg_logs[k] /= count
            if wb.get("enabled", True):
                wandb.log(avg_logs, step=epoch, commit=False)

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
            f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f} | time={time.time() - t0:.2f}s"
        )

    # ---- Optional PyTorch linear classifier on learned representations ----
    if "torch_clf" in cfg and cfg.torch_clf.get("enabled", False):
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
        optimizer = torch.optim.Adam(
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


if __name__ == "__main__":
    train_once()
