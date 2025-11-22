# train_pl_wandb.py
import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
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

    # ---------- Save final artifacts locally ----------
    out_dir = os.getcwd()  # hydra's run dir
    ckpt_path = os.path.join(out_dir, "final.ckpt")
    weights_path = os.path.join(out_dir, "model_state_dict.pt")

    trainer.save_checkpoint(ckpt_path)
    torch.save(model.state_dict(), weights_path)

    # ---------- Log as a W&B model artifact ----------
    try:
        artifact = wandb.Artifact(
            name=f"{cfg.model.name}-weights",
            type="model",
            description="Final Lightning checkpoint and PyTorch state_dict.",
            metadata={
                "model_name": cfg.model.name,
                "epochs": int(cfg.epochs),
                "seed": seed,
                "data_name": cfg.data.name,
                "trainer": {
                    "accelerator": accelerator,
                    "devices": str(devices),
                    "precision": str(precision),
                },
            },
        )
        artifact.add_file(ckpt_path)
        artifact.add_file(weights_path)
        run.log_artifact(artifact)
        logger.info("Uploaded W&B artifact.")
        os.remove(ckpt_path)
        os.remove(weights_path)
    except Exception as e:
        logger.exception(f"Failed to log W&B artifact: {e}")

    # ---------- Finish W&B run (since we created it here) ----------
    wandb.finish()


if __name__ == "__main__":
    main()
