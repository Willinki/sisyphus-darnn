from typing import TYPE_CHECKING
import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import equinox as eqx
import logging
import optax
from local_exp.models.registry import build_model
from local_exp.datasets.registry import build_dataset
from local_exp.trainers.registry import build_trainer
from local_exp.utils.evals import compute_overlaps
from local_exp.logging.logging_utils import Logger

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from darnax.trainers.interface import Trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig):
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Config: \n {OmegaConf.to_yaml(cfg, resolve=True)}")
    master_key = jax.random.key(seed=cfg.master_seed)
    wandb_logger = Logger(cfg)

    # Build model from registry
    state, orchestrator = build_model(
        cfg.model.name,
        **cfg.model.kwargs,
    )

    # build dataset from registry
    ds = build_dataset(cfg.data.name, **cfg.data.kwargs)

    # build optimizer (just adam for now)
    optimizer = optax.adam(learning_rate=cfg.optimizer.learning_rate)
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

    # build trainer
    trainer: Trainer = build_trainer(
        cfg.trainer.name,
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **cfg.trainer.kwargs,
    )

    #
    # TRAINING STARTS HERE
    #
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
        for x_b, y_b in ds.iter_eval():
            master_key, metrics = trainer.eval_step(x_b, y_b, master_key)
            accs.append(metrics["accuracy"])
            overlaps = compute_overlaps(
                trainer.orchestrator, trainer.state, x_b, y_b, master_key
            )
            overlaps_batches.append(overlaps)

        acc_eval = float(jnp.mean(jnp.array(accs)))

        # Evaluation epoch on training set
        accs = []
        for x_b, y_b in ds:
            master_key, metrics = trainer.eval_step(x_b, y_b, master_key)
            accs.append(metrics["accuracy"])

        acc_train = float(jnp.mean(jnp.array(accs)))
        logger.info(
            f"Train Accuracy = {acc_train:.3f} | Eval Accuracy = {acc_eval:.3f}"
            f" | epoch time: {time.time() - t0:.2f}s"
        )

        # logging
        wandb_logger.log_metrics(
            {"accuracy/train": acc_train, "accuracy/eval": acc_eval}, step=epoch
        )
        wandb_logger.log_weights(orchestrator, step=epoch)
        wandb_logger.log_overlaps_histograms(overlaps_batches, step=epoch)
        wandb_logger.commit(step=epoch)

    wandb_logger.finish()


if __name__ == "__main__":
    main()
