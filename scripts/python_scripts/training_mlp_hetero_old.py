# train_2phase_pl_wandb.py
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

from local_exp.models.mlp_hetero import (
    LitMLPPrototypes,
    LitPerceptronMargin,
    ModelConfig,
    OptimConfig,
)

logger = logging.getLogger(__name__)


def build_clipped_mlp_prototypes(
    input_dim,
    hidden_dim,
    output_dim,
    gain,  # kept for consistency with other builders
    lr=1e-3,
    optim="sgd",
    use_bias=True,
):
    model_cfg = ModelConfig(
        layer_sizes=[input_dim, hidden_dim, hidden_dim, output_dim],
        activation_gain=gain,
        binarize_activations=True,
        dropout=0.0,
        use_bias=use_bias,
        use_clipped_layers=True,
        loss_type="cross_entropy",  # unused here
        num_classes=output_dim,
    )
    optim_cfg = OptimConfig(name=optim, lr=lr, weight_decay=0.0)
    return LitMLPPrototypes(model_cfg, optim_cfg), None


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="binary_mlp_hetero_entangled_mnist",
)
def main(cfg: DictConfig):
    # ---------- logging + config echo ----------
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # ---------- W&B init ----------
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
    pl_logger = WandbLogger(experiment=run)

    # ---------- Reproducibility ----------
    seed = int(cfg.master_seed)
    pl.seed_everything(seed, workers=True)

    # ---------- Data ----------
    ds = build_dataset(cfg.data.name, **cfg.data.kwargs)
    data = make_lightning_datamodule_from_jax(ds, seed=seed)

    # ---------- Common trainer settings ----------
    accelerator = getattr(cfg, "accelerator", "auto")
    devices = getattr(cfg, "devices", "auto")
    precision = getattr(cfg, "precision", "32-true")
    log_every_n_steps = getattr(cfg, "log_every_n_steps", 50)
    val_check_interval = getattr(cfg, "val_check_interval", 1.0)

    # ============================================================
    # Phase 1: prototype regression on last hidden layer
    # ============================================================
    logger.info("Building phase 1 (prototype) model")
    assert cfg.model.name == "clipped-3layer-mlp-prototypes", "found {cfg.model.name}"
    proto_model, _ = build_clipped_mlp_prototypes(**cfg.model.kwargs)

    lr_monitor_1 = LearningRateMonitor(logging_interval="step")
    ckpt_cb_1 = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename="phase1_last",
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )

    trainer1 = pl.Trainer(
        max_epochs=int(cfg.epochs_phase1),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=pl_logger,
        callbacks=[lr_monitor_1, ckpt_cb_1],
        enable_progress_bar=True,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )

    logger.info("Starting phase 1 training")
    trainer1.fit(proto_model, datamodule=data)

    # ---------- Save phase 1 artifacts locally ----------
    out_dir = os.getcwd()
    ckpt_path_1 = os.path.join(out_dir, "phase1_final.ckpt")
    weights_path_1 = os.path.join(out_dir, "phase1_model_state_dict.pt")

    trainer1.save_checkpoint(ckpt_path_1)
    torch.save(proto_model.state_dict(), weights_path_1)

    # ============================================================
    # Phase 2: perceptron with margin on frozen representations
    # ============================================================
    logger.info("Building phase 2 (perceptron) model")

    # backbone is the MLP inside the phase-1 LightningModule
    backbone = proto_model.model
    rep_dim = proto_model.rep_dim
    num_classes = proto_model.num_classes

    perceptron_margin = float(cfg.perceptron.margin)
    perceptron_lr = float(cfg.perceptron.lr)

    perceptron_model = LitPerceptronMargin(
        backbone=backbone,
        rep_dim=rep_dim,
        num_classes=num_classes,
        margin=perceptron_margin,
        lr=perceptron_lr,
    )

    lr_monitor_2 = LearningRateMonitor(logging_interval="step")
    ckpt_cb_2 = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename="phase2_last",
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )

    trainer2 = pl.Trainer(
        max_epochs=int(cfg.epochs_phase2),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=pl_logger,
        callbacks=[lr_monitor_2, ckpt_cb_2],
        enable_progress_bar=True,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )

    logger.info("Starting phase 2 training")
    trainer2.fit(perceptron_model, datamodule=data)

    # ---------- Save phase 2 artifacts locally ----------
    ckpt_path_2 = os.path.join(out_dir, "phase2_final.ckpt")
    weights_path_2 = os.path.join(out_dir, "phase2_perceptron_state_dict.pt")

    trainer2.save_checkpoint(ckpt_path_2)
    torch.save(
        {
            "W": perceptron_model.W.detach().cpu(),
            "b": perceptron_model.b.detach().cpu(),
        },
        weights_path_2,
    )

    # ---------- Log both phases as a W&B model artifact ----------
    try:
        artifact = wandb.Artifact(
            name=f"{cfg.model_phase1.name}-2phase-weights",
            type="model",
            description="Two-phase training: prototype MLP (phase1) + perceptron (phase2).",
            metadata={
                "model_phase1_name": cfg.model_phase1.name,
                "epochs_phase1": int(cfg.epochs_phase1),
                "epochs_phase2": int(cfg.epochs_phase2),
                "seed": seed,
                "data_name": cfg.data.name,
                "trainer": {
                    "accelerator": accelerator,
                    "devices": str(devices),
                    "precision": str(precision),
                },
                "perceptron": {
                    "margin": perceptron_margin,
                    "lr": perceptron_lr,
                },
            },
        )
        artifact.add_file(ckpt_path_1)
        artifact.add_file(weights_path_1)
        artifact.add_file(ckpt_path_2)
        artifact.add_file(weights_path_2)
        run.log_artifact(artifact)
        logger.info("Uploaded W&B artifact for 2-phase training.")
        os.remove(ckpt_path_1)
        os.remove(weights_path_1)
        os.remove(ckpt_path_2)
        os.remove(weights_path_2)
    except Exception as e:
        logger.exception(f"Failed to log W&B artifact: {e}")

    # ---------- Finish W&B run ----------
    wandb.finish()


if __name__ == "__main__":
    main()
