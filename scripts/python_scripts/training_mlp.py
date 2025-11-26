# train_pl_wandb.py
import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from local_exp.datasets.jax_to_lightningdata import make_lightning_datamodule_from_jax
from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="binary_mlp")
def main(cfg: DictConfig):
    # ---------- logging + config echo ----------
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # ---------- W&B init (no custom Logger; do it here) ----------
    logger.info("Initializing Weights & Biases")
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
        save_code=True,
        reinit=False,
    )
    # Wire PL logger to the active run
    pl_logger = WandbLogger(experiment=run)

    # ---------- Reproducibility ----------
    seed = int(cfg.master_seed)
    pl.seed_everything(seed, workers=True)

    # ---------- Build model (LightningModule) ----------
    model, _ = build_model(cfg.model.name, **cfg.model.kwargs)

    # ---------- Build data (LightningDataModule) ----------
    ds = build_dataset(cfg.data.name, **cfg.data.kwargs)
    data = make_lightning_datamodule_from_jax(ds, seed=seed)

    # ---------- Callbacks ----------
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        save_top_k=0,  # we manually save a final checkpoint at the end
        save_last=True,  # keep last.ckpt for convenience
        filename="last",
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )

    # ---------- Trainer ----------
    accelerator = getattr(cfg, "accelerator", "auto")
    devices = getattr(cfg, "devices", "auto")
    precision = getattr(cfg, "precision", "32-true")
    log_every_n_steps = getattr(cfg, "log_every_n_steps", 50)
    val_check_interval = getattr(cfg, "val_check_interval", 1.0)

    trainer = pl.Trainer(
        max_epochs=int(cfg.epochs),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=pl_logger,
        callbacks=[lr_monitor, ckpt_cb],
        enable_progress_bar=True,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )

    # ---------- Train ----------
    trainer.fit(model, datamodule=data)

    if "torch_clf" in cfg and cfg.torch_clf.get("enabled", False):
        torch_clf_cfg = cfg.torch_clf
        print("Starting PyTorch linear classifier training...")
        # Seed
        if "master_seed" in cfg:
            torch.manual_seed(int(cfg.master_seed))

        device = model.device
        model.model.eval()

        # Feature extraction: TRAIN split
        train_reps = []
        train_labels = []
        for xb, yb in data.train_dataloader():
            xb_t = xb.to(device)
            reps = model.model.forward_features(xb_t).detach().cpu()
            train_reps.append(reps)
            yb_idx = torch.argmax(yb, dim=-1)
            train_labels.append(yb_idx.detach().cpu())

        # Feature extraction: EVAL split
        test_reps = []
        test_labels = []
        test_loader = data.val_dataloader()
        for xb, yb in test_loader:
            xb_t = xb.to(device)
            reps = model.model.forward_features(xb_t).detach().cpu()
            test_reps.append(reps)
            yb_idx = torch.argmax(yb, dim=-1)
            test_labels.append(yb_idx.detach().cpu())

        features_train = torch.cat(train_reps, dim=0).float()
        labels_train = torch.cat(train_labels, dim=0).long()
        features_test = torch.cat(test_reps, dim=0).float()
        labels_test = torch.cat(test_labels, dim=0).long()

        input_dim = int(features_train.shape[1])
        num_classes = int(cfg.data.num_labels)

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
                    commit=True,
                )

            print(
                f"[Torch Clf] Epoch {e:03d} | train_acc={train_acc:.4f} | eval_acc={eval_acc:.4f} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}"
            )

        print("PyTorch linear classifier training complete.")

    # ---------- Finish W&B run (since we created it here) ----------
    wandb.finish()


if __name__ == "__main__":
    main()
