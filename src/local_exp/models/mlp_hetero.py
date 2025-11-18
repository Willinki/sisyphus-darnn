from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from local_exp.models.registry import register_model

from .mlp_lightning import MLPClipped, ModelConfig, OptimConfig


class LitMLPPrototypes(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        optim_cfg: OptimConfig = OptimConfig(),
    ):
        super().__init__()
        self.save_hyperparameters(
            {"model_cfg": model_cfg.__dict__, "optim_cfg": optim_cfg.__dict__}
        )

        if not model_cfg.use_clipped_layers:
            raise ValueError(
                "LitMLPPrototypes assumes a clipped MLP (use_clipped_layers=True)."
            )

        # feature extractor is still the full MLPClipped, but we will use forward_features
        self.model = MLPClipped(
            layer_sizes=model_cfg.layer_sizes,
            use_bias=model_cfg.use_bias,
            dropout=model_cfg.dropout,
            binarize=model_cfg.binarize_activations,
        )

        # representation dimension = last hidden dim
        self.rep_dim = model_cfg.layer_sizes[-2]
        self.num_classes = (
            model_cfg.num_classes
            if model_cfg.num_classes is not None
            else model_cfg.layer_sizes[-1]
        )

        if self.num_classes > self.rep_dim:
            raise ValueError(
                f"Need rep_dim >= num_classes for orthonormal prototypes, "
                f"got rep_dim={self.rep_dim}, C={self.num_classes}"
            )

        # sample orthonormal prototypes: shape (C, rep_dim)
        A = torch.randn(self.rep_dim, self.num_classes)
        Q, _ = torch.linalg.qr(A)  # (rep_dim, num_classes), columns orthonormal
        prototypes = Q.T.contiguous()  # (num_classes, rep_dim)
        self.register_buffer("prototypes", prototypes)  # fixed during training

        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return representation z(x)."""
        return self.model.forward_features(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        # y can be one-hot or int; handle both
        if y.ndim == 2:
            labels = y.argmax(dim=1)
        else:
            labels = y

        z = self.model.forward_features(x)  # (B, rep_dim)
        target = self.prototypes[labels]  # (B, rep_dim)
        loss = self.criterion(z, target)

        # "classification" via nearest prototype to monitor progress
        with torch.no_grad():
            logits = z @ self.prototypes.T  # (B, C)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        oc = (
            OptimConfig(**self.hparams["optim_cfg"])
            if isinstance(self.hparams["optim_cfg"], dict)
            else self.hparams["optim_cfg"]
        )
        params = self.parameters()
        if oc.name.lower() == "adam":
            return torch.optim.Adam(
                params, lr=oc.lr, betas=oc.betas, weight_decay=oc.weight_decay
            )
        elif oc.name.lower() == "sgd":
            return torch.optim.SGD(
                params, lr=oc.lr, momentum=oc.momentum, weight_decay=oc.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {oc.name}")


class LitPerceptronMargin(pl.LightningModule):
    def __init__(
        self,
        backbone: MLPClipped,
        rep_dim: int,
        num_classes: int,
        margin: float = 1.0,
        lr: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.rep_dim = rep_dim
        self.num_classes = num_classes
        self.margin = margin
        self.lr = lr

        # we will update W, b manually -> no optimizer, no autograd needed
        self.W = nn.Parameter(torch.zeros(num_classes, rep_dim))
        # self.b = nn.Parameter(torch.zeros(num_classes))

        self.automatic_optimization = False  # manual updates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone.forward_features(x)
        scores = F.linear(z, self.W, None)  # (B, C)
        return scores

    @torch.no_grad()
    def perceptron_update(self, z: torch.Tensor, labels: torch.Tensor):
        """
        z: (B, D) frozen representations
        labels: (B,) class indices
        """
        scores = F.linear(z, self.W, self.b)  # (B, C)
        B, C = scores.shape

        target_scores = scores[torch.arange(B), labels]

        # best competitor (max over k != y)
        neg_scores = scores.clone()
        neg_scores[torch.arange(B), labels] = -1e9
        max_neg_scores, max_neg_idx = neg_scores.max(dim=1)

        # margin violations
        violations = target_scores <= max_neg_scores + self.margin
        if not violations.any():
            return

        z_v = z[violations]  # (Bv, D)
        y_v = labels[violations]  # (Bv,)
        j_v = max_neg_idx[violations]  # (Bv,)

        dW = torch.zeros_like(self.W)
        # db = torch.zeros_like(self.b)

        # W_y += z, W_j -= z
        dW.index_add_(0, y_v, z_v)
        dW.index_add_(0, j_v, -z_v)

        # # b_y += 1, b_j -= 1
        # ones = torch.ones_like(y_v, dtype=db.dtype)
        # db.index_add_(0, y_v, ones)
        # db.index_add_(0, j_v, -ones)

        self.W.data += self.lr * dW
        # self.b.data += self.lr * db

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        labels = y.argmax(dim=1) if y.ndim == 2 else y

        with torch.no_grad():
            z = self.backbone.forward_features(x)

        self.perceptron_update(z, labels)

        # log training accuracy
        with torch.no_grad():
            scores = F.linear(z, self.W, self.b)
            preds = scores.argmax(dim=1)
            acc = (preds == labels).float().mean()

        self.log("train/acc", acc, prog_bar=True, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        labels = y.argmax(dim=1) if y.ndim == 2 else y
        with torch.no_grad():
            z = self.backbone.forward_features(x)
            scores = F.linear(z, self.W, self.b)
            preds = scores.argmax(dim=1)
            acc = (preds == labels).float().mean()
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        labels = y.argmax(dim=1) if y.ndim == 2 else y
        with torch.no_grad():
            z = self.backbone.forward_features(x)
            scores = F.linear(z, self.W, self.b)
            preds = scores.argmax(dim=1)
            acc = (preds == labels).float().mean()
        self.log("test/acc", acc, prog_bar=True, on_epoch=True, logger=True)

    def configure_optimizers(self):
        # no optimizer: updates are done manually in perceptron_update
        return []
