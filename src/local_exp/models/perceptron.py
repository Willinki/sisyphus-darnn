from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from local_exp.models.registry import register_model
from local_exp.models.mlp_lightning import BinaryActivationSTE


# -----------------------------
#  Fixed random projection -> sign -> linear
# -----------------------------
class FixedRandomProjection(nn.Module):
    """Fixed (non-trainable) random projection with N(0, 1/sqrt(in_features))."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        W = torch.randn(out_features, in_features, generator=gen) / (in_features**0.5)
        self.register_buffer(
            "weight", W, persistent=False
        )  # not trainable, moves with .to()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()  # [B, in] @ [in, out]^T -> [B, out]


class RandomProjectionSignLinear(nn.Module):
    """Fixed RP -> sign (±1 via STE) -> trainable linear head."""

    def __init__(
        self,
        in_features: int,
        proj_features: int,
        out_features: int,
        rp_seed: int = 0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.rp = FixedRandomProjection(in_features, proj_features, seed=rp_seed)
        self.sign = BinaryActivationSTE()
        self.head = nn.Linear(proj_features, out_features, bias=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.head.weight)  # perceptron often from zero
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def featurize(self, x: torch.Tensor) -> torch.Tensor:
        return self.sign(self.rp(x))  # ±1 features (STE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.featurize(x))


# -----------------------------
#  Lightning Module (expects y ∈ {−1,+1} with shape [B, C])
# -----------------------------
@dataclass
class PerceptronConfig:
    in_features: int
    proj_features: int
    out_features: int
    margin: float = 1.0
    lr: float = 1.0
    rp_seed: int = 0
    use_bias: bool = True


class LitRandProjPerceptron(pl.LightningModule):
    """
    Random projection (fixed) -> sign (STE) -> linear head,
    trained with batch-summed perceptron-with-margin:

        ΔW_ij = Σ_b 1[ŷ_{b,j} * y_{b,j} < m] * y_{b,j} * x_{b,i}
        Δb_j  = Σ_b 1[ŷ_{b,j} * y_{b,j} < m] * y_{b,j}

    where y ∈ {−1,+1}^{B×C} (already provided by the dataloader),
    and x_b are the signed features after RP->sign.
    """

    def __init__(self, cfg: PerceptronConfig):
        super().__init__()
        self.save_hyperparameters({"perceptron_cfg": cfg.__dict__})
        self.model = RandomProjectionSignLinear(
            cfg.in_features,
            cfg.proj_features,
            cfg.out_features,
            cfg.rp_seed,
            cfg.use_bias,
        )
        self.margin = float(cfg.margin)
        self.lr = float(cfg.lr)
        self.automatic_optimization = False  # manual perceptron updates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def _perceptron_update(self, phi: torch.Tensor, y_pm1: torch.Tensor):
        """
        y_pm1: [B, C] with entries in {−1,+1}
        phi:   [B, D] signed features after RP->sign
        """
        # Scores under current head
        scores = F.linear(phi, self.model.head.weight, self.model.head.bias)  # [B, C]
        y_pm1 = y_pm1.to(scores.dtype)

        # Violations: elementwise (score * label) < margin
        violate = (scores * y_pm1) < self.margin  # [B, C] bool
        M = violate.to(scores.dtype) * y_pm1  # [B, C] in {0, ±1}

        # Batch-summed perceptron updates
        dW = M.t() @ phi  # [C, D]
        self.model.head.weight.data += self.lr * dW

        if self.model.head.bias is not None:
            db = M.sum(dim=0)  # [C]
            self.model.head.bias.data += self.lr * db

        # Metrics
        loss = F.relu(self.margin - (scores * y_pm1)).mean()
        preds = scores.argmax(dim=1)
        labels = y_pm1.argmax(dim=1)  # since true class has +1 and others −1
        acc = (preds == labels).float().mean()
        return loss, acc

    def _shared_step(self, batch, stage: str):
        x, y_pm1 = batch  # y already in {−1,+1} with shape [B, C]
        # Featurize once
        with torch.no_grad():
            phi = self.model.featurize(x)

        if stage == "train":
            loss, acc = self._perceptron_update(phi, y_pm1)
        else:
            with torch.no_grad():
                scores = F.linear(phi, self.model.head.weight, self.model.head.bias)
                y_pm1 = y_pm1.to(scores.dtype)
                loss = F.relu(self.margin - (scores * y_pm1)).mean()
                preds = scores.argmax(dim=1)
                labels = y_pm1.argmax(dim=1)
                acc = (preds == labels).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, logger=True)
        return loss if stage == "train" else None

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # No optimizer needed; manual perceptron updates.
        return torch.optim.SGD(self.parameters(), lr=0.0)


# -----------------------------
#  Registry builder
# -----------------------------
@register_model("rp-sign-perceptron")
def build_rp_sign_perceptron(
    input_dim: int,
    proj_dim: int,
    output_dim: int,
    margin: float = 1.0,
    lr: float = 1.0,
    rp_seed: int = 0,
    use_bias: bool = True,
    **kwargs,
):
    cfg = PerceptronConfig(
        in_features=input_dim,
        proj_features=proj_dim,
        out_features=output_dim,
        margin=margin,
        lr=lr,
        rp_seed=rp_seed,
        use_bias=use_bias,
    )
    return LitRandProjPerceptron(cfg), None
