# search_optuna_asha.py  (Ray-free single run)
import copy
import equinox as eqx
import jax
from local_exp.models.registry import build_model

kwargs = {
    "seed": 99,
    "dim_data": 100,
    "dim_hidden": 100,
    "num_labels": 10,
    "strength_forth": 5.0,
    "strength_back": 1.4,
    "threshold_j": 1.4,
    "threshold_in": 1.4,
    "threshold_out": 3.0,
    "j_d": 0.5,
}
state, orchestrator = build_model("fc-baseline", **kwargs)
wback = orchestrator.lmap[1][2].W
wout = orchestrator.lmap[2][1].W
print("wout shape:", wout.shape)
print("wback shape:", wback.shape)
print("wout norm:", jax.numpy.linalg.norm(wout))
print("wback norm:", jax.numpy.linalg.norm(wback))
C, H = wback.shape[0], wback.shape[1]
assert wout.shape[0] == H and wout.shape[1] == C
scale_ratio = (C / H) ** 0.5 / kwargs["strength_back"]
rescaled_transpose_wback = copy.deepcopy(wback).T * scale_ratio
orchestrator = eqx.tree_at(
    lambda x: x.lmap[2][1].W, orchestrator, rescaled_transpose_wback
)
print("wout shape after:", orchestrator.lmap[2][1].W.shape)
print("wback shape after:", orchestrator.lmap[1][2].W.shape)
print("wout norm after:", jax.numpy.linalg.norm(orchestrator.lmap[2][1].W))
print("wback norm after:", jax.numpy.linalg.norm(orchestrator.lmap[1][2].W))
