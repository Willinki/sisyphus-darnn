"""
multi_logit_margin_mlp.py

Standalone model:
- Hidden layers: Linear + sign (STE), no clamping
- Output head: Linear logits h (no binarization)
- Loss: Per-logit margin (one-vs-all) L = mean ReLU(kappa - y * h),
        where y \in {-1, +1}^C (typically one +1 for the true class).

Suggested usage:
- Provide labels as +/-1 matrices (B, C) with one +1 per sample.
- Keep reduction='mean' for LR stability across batch sizes / class counts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# If you use the project's registry, keep this import; otherwise you can remove it
try:
    from local_exp.models.registry import register_model
except Exception:  # pragma: no cover

    def register_model(name):  # fallback no-op decorator
        def deco(fn):
            return fn

        return deco


# -----------------------------
#  Optim config
# -----------------------------
@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


# -----------------------------
#  Binary (STE) activation
# -----------------------------
class _StraightThroughSignFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.sign(x)
        # map zeros to +1 to avoid ambiguity
        y = torch.where(y == 0, torch.ones_like(y), y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # identity pass-through (classic STE)
        return grad_output


class BinaryActivationSTE(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _StraightThroughSignFn.apply(x)


# -----------------------------
#  Layers & MLP
# -----------------------------
class LinearBinary(nn.Module):
    """Linear layer followed by straight-through sign binarization (no clamp)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.binarize = BinaryActivationSTE()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binarize(self.linear(x))


class MLPBinaryNoClamp(nn.Module):
    """
    Hidden layers: LinearBinary (Linear + sign STE)
    Output layer: plain Linear (raw logits h)
    """

    def __init__(
        self, layer_sizes: List[int], use_bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        blocks = []
        L = len(layer_sizes) - 1
        for i in range(L):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            is_last = i == L - 1
            if not is_last:
                blocks.append(LinearBinary(in_f, out_f, bias=use_bias))
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            else:
                blocks.append(nn.Linear(in_f, out_f, bias=use_bias))  # logits h
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
#  Loss: per-logit margin (one-vs-all)
# -----------------------------
class MultiLogitMarginLoss(nn.Module):
    def __init__(self, kappa: float = 1.0, reduction: str = "batch_mean_class_sum"):
        super().__init__()
        assert reduction in ("mean", "sum", "none", "batch_mean_class_sum")
        self.kappa = kappa
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y_pm1: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), y_pm1: (B, C) in {-1,+1}
        if y_pm1.dtype != logits.dtype:
            y_pm1 = y_pm1.to(dtype=logits.dtype)
        losses = F.relu(self.kappa - y_pm1 * logits)  # (B, C)
        if self.reduction == "mean":
            return losses.mean()  # mean over B*C
        elif self.reduction == "sum":
            return losses.sum()  # sum over B*C
        elif self.reduction == "batch_mean_class_sum":
            return losses.sum(dim=1).mean()  # sum over C, mean over B
        else:  # "none"
            return losses  # (B, C)


# -----------------------------
#  Lightning Module
# -----------------------------
class MultiLogitMarginMLP(pl.LightningModule):
    """
    Expects batches (x, y_pm1) where y_pm1 is a +/-1 matrix of shape (B, C)
    with exactly one +1 per row (standard multiclass encoded as pm1).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        kappa: float = 1.0,
        dropout: float = 0.0,
        use_bias: bool = True,
        reduction: str = "mean",
        optim_cfg: OptimConfig = OptimConfig(),
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "layer_sizes": layer_sizes,
                "kappa": kappa,
                "dropout": dropout,
                "use_bias": use_bias,
                "reduction": reduction,
                "optim_cfg": (
                    optim_cfg.__dict__
                    if isinstance(optim_cfg, OptimConfig)
                    else optim_cfg
                ),
            }
        )
        self.model = MLPBinaryNoClamp(
            layer_sizes=layer_sizes, use_bias=use_bias, dropout=dropout
        )
        self.criterion = MultiLogitMarginLoss(kappa=kappa, reduction=reduction)

    # ---- Forward returns raw logits h ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---- Steps ----
    def _shared_step(self, batch, stage: str):
        x, y_pm1 = batch  # y_pm1: (B, C) in {-1,+1}
        logits = self(x)  # raw logits h
        loss = self.criterion(logits, y_pm1)

        with torch.no_grad():
            # derive class indices from targets and logits
            target_idx = y_pm1.argmax(dim=1)
            preds_idx = logits.argmax(dim=1)
            acc = (preds_idx == target_idx).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    # ---- Optimizer ----
    def configure_optimizers(self):
        oc = (
            OptimConfig(**self.hparams["optim_cfg"])
            if isinstance(self.hparams["optim_cfg"], dict)
            else self.hparams["optim_cfg"]
        )
        params = self.parameters()
        opt_name = oc.name.lower()
        if opt_name == "adam":
            return torch.optim.Adam(
                params, lr=oc.lr, betas=oc.betas, weight_decay=oc.weight_decay
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                params, lr=oc.lr, momentum=oc.momentum, weight_decay=oc.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {oc.name}")


# -----------------------------
#  Builder (registry optional)
# -----------------------------
@register_model("multi-logit-margin-mlp")
def build_multi_logit_margin_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    kappa: float = 1.0,
    lr: float = 1e-3,
    optim: str = "sgd",
    dropout: float = 0.0,
    use_bias: bool = False,
    reduction: str = "batch_mean_class_sum",
):
    """
    Constructs a 3-layer MLP:
      [input_dim] -> [hidden_dim] -> [hidden_dim] -> [output_dim]
    Hidden layers use sign-STE; head is linear logits.
    Loss: per-logit margin (one-vs-all) with given kappa and reduction.
    """
    layer_sizes = [input_dim, hidden_dim, hidden_dim, output_dim]
    optim_cfg = OptimConfig(name=optim, lr=lr, weight_decay=0.0)
    model = MultiLogitMarginMLP(
        layer_sizes=layer_sizes,
        kappa=kappa,
        dropout=dropout,
        use_bias=use_bias,
        reduction=reduction,
        optim_cfg=optim_cfg,
    )
    return model, None
