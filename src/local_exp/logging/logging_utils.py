import logging
from jax import Array
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.layer_maps.sparse import LayerMap


logger = logging.getLogger(__name__)


class Logger:
    def __init__(self, cfg):
        self.enabled = cfg.wandb.enabled
        self.wandb = None

        if self.enabled:
            logger.info("Initializing wandb logger")
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                mode=cfg.wandb.mode,
                dir=cfg.wandb.dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=list(cfg.wandb.tags),
                save_code=True,
                reinit=False,
            )
            self.wandb = wandb

    def log_weights(self, orchestrator: SequentialOrchestrator, step: int):
        if not self.enabled:
            return
        logger.info("Logging model weights")
        lmap: LayerMap = orchestrator.lmap
        for i, senders_group in lmap.row_items():
            for j, mod in senders_group.items():
                try:
                    if i == j:
                        weights = mod.J
                    else:
                        weights = mod.W
                    self.wandb.log(
                        {f"weights/{j}-to-{i}": wandb.Histogram(weights)},
                        step=step,
                        commit=False,
                    )
                except AttributeError as e:
                    logger.info(f"{e} - SKIPPING")

    def log_metrics(self, metrics: dict[str, float], step: int):
        if not self.enabled:
            return
        logger.info("Logging metrics")
        self.wandb.log(metrics, step=step, commit=False)

    def log_overlaps_histograms(self, overlaps: list[list[Array | None]], step: int):
        if not self.enabled:
            return
        logger.info("Logging overlaps histogram")
        logger.warning("Keeping only layer one")
        overlaps = jnp.concat([overlap[1] for overlap in overlaps])
        different_overlaps_perc = jnp.mean(overlaps < 0.99)
        overlaps_histograms = {"overlaps/prime_star_1": wandb.Histogram(overlaps)}
        wandb.log(
            overlaps_histograms
            | {"overlaps/prime_star_is_different_ratio": different_overlaps_perc},
            step=step,
            commit=False,
        )

    def commit(self, step: int):
        if not self.enabled:
            return
        logger.info("Committing logs")
        self.wandb.log({"step": step}, step=step, commit=True)

    def finish(self):
        if not self.enabled:
            return
        logger.info("finishing run")
        self.wandb.finish()
