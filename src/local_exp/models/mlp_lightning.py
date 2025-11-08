from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from local_exp.models.registry import register_model


# -----------------------------
#  Activations & Layers
# -----------------------------
class TanhWithGain(nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = float(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gain * x)


class _StraightThroughSignFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.sign(x)
        y = torch.where(y == 0, torch.ones_like(y), y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class BinaryActivationSTE(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _StraightThroughSignFn.apply(x)


class TanhThenMaybeBinarize(nn.Module):
    def __init__(self, gain: float = 1.0, binarize: bool = False):
        super().__init__()
        self.tanh = TanhWithGain(gain)
        self.binarize = BinaryActivationSTE() if binarize else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(x)
        if self.binarize is not None:
            x = self.binarize(x)
        return x


class LinearClipped(nn.Module):
    """Linear layer followed by clipping to [-1, 1].
    Optionally binarize outputs using a straight-through estimator after clipping.
    Values beyond ±1 are saturated; gradients through the clip are zero outside the range,
    while the STE provides passthrough gradients for the binarization step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        binarize: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.binarize = BinaryActivationSTE() if binarize else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(self.linear(x), -1.0, 1.0)
        if self.binarize is not None:
            x = self.binarize(x)
        return x


# -----------------------------
#  MLP Builders
# -----------------------------
class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        activation_gain: float = 1.0,
        binarize_activations: bool = False,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        blocks = []
        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            is_last = i == len(layer_sizes) - 2
            blocks.append(nn.Linear(in_f, out_f, bias=use_bias))
            if not is_last:
                blocks.append(
                    TanhThenMaybeBinarize(activation_gain, binarize_activations)
                )
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClipped(nn.Module):
    """Alternative MLP variant using LinearClipped layers instead of tanh/binary ones."""

    def __init__(
        self,
        layer_sizes: List[int],
        use_bias: bool = True,
        dropout: float = 0.0,
        binarize: bool = True,
    ):
        super().__init__()
        blocks = []
        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            is_last = i == len(layer_sizes) - 2
            blocks.append(LinearClipped(in_f, out_f, bias=use_bias, binarize=binarize))
            if not is_last and dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
#  Losses
# -----------------------------
class MultiClassMarginArgMaxLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        mask = F.one_hot(targets, num_classes=C).bool()
        neg_logits = logits.masked_fill(mask, float("-inf"))
        max_neg = neg_logits.max(dim=1).values
        margins = true_logits - max_neg
        loss = F.relu(self.margin - margins)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
#  Lightning Module
# -----------------------------
@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Optional[tuple] = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class ModelConfig:
    layer_sizes: List[int] = None
    activation_gain: float = 1.0
    binarize_activations: bool = False
    dropout: float = 0.0
    use_bias: bool = True
    use_clipped_layers: bool = False  # NEW
    loss_type: str = "cross_entropy"
    argmax_margin: float = 1.0
    num_classes: Optional[int] = None


class LitMLP(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, optim_cfg: OptimConfig = OptimConfig()):
        super().__init__()
        self.save_hyperparameters(
            {"model_cfg": model_cfg.__dict__, "optim_cfg": optim_cfg.__dict__}
        )

        if model_cfg.use_clipped_layers:
            self.model = MLPClipped(
                layer_sizes=model_cfg.layer_sizes,
                use_bias=model_cfg.use_bias,
                dropout=model_cfg.dropout,
                binarize=model_cfg.binarize_activations,
            )
        else:
            self.model = MLP(
                layer_sizes=model_cfg.layer_sizes,
                activation_gain=model_cfg.activation_gain,
                binarize_activations=model_cfg.binarize_activations,
                use_bias=model_cfg.use_bias,
                dropout=model_cfg.dropout,
            )

        if model_cfg.loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif model_cfg.loss_type == "argmax_margin":
            self.criterion = MultiClassMarginArgMaxLoss(margin=model_cfg.argmax_margin)
        else:
            raise ValueError(f"Unknown loss_type: {model_cfg.loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        label = y.argmax(dim=1)
        logits = self(x)
        loss = self.criterion(logits, label)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == label).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log(f"{stage}/acc", acc, on_epoch=True, prog_bar=True, logger=True)
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


@register_model("tanh-3layer-mlp")
def build_tanh_mlp(
    input_dim, hidden_dim, output_dim, gain, loss_type="cross_entropy", lr=1e-3
):
    model_cfg = ModelConfig(
        layer_sizes=[input_dim, hidden_dim, hidden_dim, output_dim],
        activation_gain=gain,
        binarize_activations=False,
        dropout=0.0,
        use_bias=True,
        use_clipped_layers=False,
        loss_type=loss_type,
    )
    optim_cfg = OptimConfig(name="adam", lr=lr, weight_decay=0.0)
    return LitMLP(model_cfg, optim_cfg), None


@register_model("binary-3layer-mlp")
def build_binary_mlp(
    input_dim, hidden_dim, output_dim, gain, loss_type="cross_entropy", lr=1e-3
):
    model_cfg = ModelConfig(
        layer_sizes=[input_dim, hidden_dim, hidden_dim, output_dim],
        activation_gain=gain,
        binarize_activations=True,
        dropout=0.0,
        use_bias=True,
        use_clipped_layers=False,
        loss_type=loss_type,
    )
    optim_cfg = OptimConfig(name="adam", lr=lr, weight_decay=0.0)
    return LitMLP(model_cfg, optim_cfg), None


@register_model("clipped-3layer-mlp")
def build_clipped_mlp(
    input_dim, hidden_dim, output_dim, gain, loss_type="cross_entropy", lr=1e-3
):
    model_cfg = ModelConfig(
        layer_sizes=[input_dim, hidden_dim, hidden_dim, output_dim],
        activation_gain=gain,  # Note: unused in MLPClipped, but kept for consistency
        binarize_activations=True,
        dropout=0.0,
        use_bias=True,
        use_clipped_layers=True,
        loss_type=loss_type,
    )
    optim_cfg = OptimConfig(name="adam", lr=lr, weight_decay=0.0)
    return LitMLP(model_cfg, optim_cfg), None
