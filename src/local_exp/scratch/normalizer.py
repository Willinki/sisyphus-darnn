from __future__ import annotations
from typing import Any, Dict, Tuple
import jax.numpy as jnp
import equinox as eqx

from darnax.modules.fully_connected import FullyConnected, FrozenFullyConnected
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator

Index = Tuple[int, int]  # (i, j)


def _vector_or_frob_norm(mat: jnp.ndarray, axis: int | None) -> jnp.ndarray:
    if axis is None:
        return jnp.linalg.norm(mat)  # Frobenius
    if axis in (0, 1):
        return jnp.linalg.vector_norm(mat, ord=2, axis=axis)
    raise ValueError("axis must be one of {None, 0, 1}.")


def _safe_scale(saved: jnp.ndarray, current: jnp.ndarray, eps: float) -> jnp.ndarray:
    zero_saved = jnp.equal(saved, 0)
    good = jnp.greater(current, eps)
    base = jnp.where(good, saved / current, 1.0)
    return jnp.where(~good & zero_saved, 0.0, base)


def save_norms(orchestrator: Any, axis: int | None = None) -> Dict[Index, jnp.ndarray]:
    """
    Read-only pass. Collect a baseline norm for each module.
    Keys are (i, j) indices into orchestrator.lmap.
    """
    norms: Dict[Index, jnp.ndarray] = {}
    for i, row in orchestrator.lmap.row_items():
        for j, mod in row.items():
            if isinstance(mod, FullyConnected):
                norms[(i, j)] = _vector_or_frob_norm((mod.W), axis)
            elif isinstance(mod, RecurrentDiscrete):
                J_off = (mod.J) * (mod._mask)  # off-diagonal only
                norms[(i, j)] = _vector_or_frob_norm(J_off, axis)
            # else: ignore
    return norms


def normalize(
    orchestrator: Any,
    norms: Dict[Index, jnp.ndarray],
    axis: int | None = None,
    eps: float = 1e-5,
) -> Any:
    """
    Return a new orchestrator with weights rescaled to match `norms`.
    Only uses eqx.tree_at on leaf arrays; does not rebuild lmap containers by hand.
    """
    new_orch = orchestrator
    for (i, j), target in norms.items():
        mod = new_orch.lmap[i][j]

        if isinstance(mod, FullyConnected):
            W = mod.W
            curr = _vector_or_frob_norm(W, axis)
            scale = _safe_scale(target, curr, eps)
            if axis is None:
                W_new = W * scale
            elif axis == 0:
                W_new = W * scale.reshape((1, -1))
            elif axis == 1:
                W_new = W * scale.reshape((-1, 1))
            else:
                raise ValueError("axis must be one of {None, 0, 1}.")

            new_orch = eqx.tree_at(lambda o: o.lmap[i][j].W, new_orch, W_new)

        elif isinstance(mod, RecurrentDiscrete):
            J = mod.J
            mask = mod._mask  # True off-diagonal, False on diagonal
            J_off = J * mask
            curr = _vector_or_frob_norm(J_off, axis)
            scale = _safe_scale(target, curr, eps)
            if axis is None:
                J_off_scaled = J_off * scale
            elif axis == 0:
                J_off_scaled = J_off * scale.reshape((1, -1))
            elif axis == 1:
                J_off_scaled = J_off * scale.reshape((-1, 1))
            else:
                raise ValueError("axis must be one of {None, 0, 1}.")

            # keep diagonal exactly; replace only off-diagonal entries
            J_new = J_off_scaled + J * (1 - mask)
            new_orch = eqx.tree_at(lambda o: o.lmap[i][j].J, new_orch, J_new)

        # else: skip unknown module types

    return new_orch


def decay(
    orchestrator: SequentialOrchestrator,
    config: Dict[str, Any],
) -> Any:
    """
    Return a new orchestrator with exponentially decayed weights.

    Each weight matrix is multiplied by (1 - rho), simulating
    weight decay instead of hard normalization.

    Parameters
    ----------
    orchestrator : Any
        The orchestrator module containing submodules in lmap[i][j].
    rho : float
        Decay rate. Each weight is multiplied by (1 - rho).

    Returns
    -------
    Any
        A new orchestrator with decayed weights.
    """
    # NOTE: here, we are using config value of learning rate; however, with sparsity, this is not the actual lr used.
    # This is not a bug, but it makes interpreting the weight decay harder...
    # NOTE: compared with the old codebase, we are not scaling the weight decay coefficient by the magnitude of the weights at init,
    # again because we read from config (e.g. strength_back, strength_in).
    new_orch = orchestrator

    # W_in
    # NOTE: with sparsity, here we also have a mask like for J. However, zero entries remain zero after decay, so we can skip it.
    W_in = new_orch.lmap[1][0].W
    rescaling_win = (
        config["optimizer"]["weight_decay_win"]
        * config["optimizer"]["learning_rate_win"]
        * jnp.sqrt(1 - 0.90)
        / (config["model"]["kwargs"]["dim_data"] ** 0.5)
    )
    W_in_new = W_in * (1.0 - rescaling_win)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][0].W, new_orch, W_in_new)

    # J
    J = new_orch.lmap[1][1].J
    mask = new_orch.lmap[1][1]._mask  # exclude diagonal from decay
    rescaling_j = (
        config["optimizer"]["weight_decay_j"]
        * config["optimizer"]["learning_rate_j"]
        * jnp.sqrt(1 - 0.99)
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    J_new = J * (1.0 - rescaling_j * mask)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][1].J, new_orch, J_new)

    # W_out
    W_out = new_orch.lmap[2][1].W
    rescaling_wout = (
        config["optimizer"]["weight_decay_wout"]
        * config["optimizer"]["learning_rate_wout"]
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    W_out_new = W_out * (1.0 - rescaling_wout)
    new_orch = eqx.tree_at(lambda o: o.lmap[2][1].W, new_orch, W_out_new)

    # W_back
    if config["model"]["kwargs"]["learnable_wback"]:
        # this would be a no-op otherwise, since lr is zero
        W_back = new_orch.lmap[1][2].W
        rescaling_wback = (
            config["optimizer"]["weight_decay_wback"]
            * config["optimizer"]["learning_rate_wback"]
            / (config["model"]["kwargs"]["num_labels"] ** 0.5)
        )
        W_back_new = W_back * (1.0 - rescaling_wback)
        new_orch = eqx.tree_at(lambda o: o.lmap[1][2].W, new_orch, W_back_new)

    return new_orch


def clip(
    orchestrator: SequentialOrchestrator,
    rho: float,
) -> Any:
    """
    Return a new orchestrator with exponentially decayed weights.

    Each weight matrix is multiplied by (1 - rho), simulating
    weight decay instead of hard normalization.

    Parameters
    ----------
    orchestrator : Any
        The orchestrator module containing submodules in lmap[i][j].
    rho : float
        Decay rate. Each weight is multiplied by (1 - rho).

    Returns
    -------
    Any
        A new orchestrator with decayed weights.
    """
    new_orch = orchestrator
    for receiver_idx, senders_group in new_orch.lmap.row_items():
        for sender_idx, mod in senders_group.items():
            mod = new_orch.lmap[receiver_idx][sender_idx]

            if isinstance(mod, FullyConnected) and not isinstance(
                mod, FrozenFullyConnected
            ):
                W = mod.W

                new_orch = eqx.tree_at(
                    lambda o: o.lmap[receiver_idx][sender_idx].W, new_orch, W_new
                )

            elif isinstance(mod, RecurrentDiscrete):
                J = mod.J
                mask = mod._mask  # exclude diagonal from decay
                J_new = J * (1.0 - rho * mask)
                new_orch = eqx.tree_at(
                    lambda o: o.lmap[receiver_idx][sender_idx].J, new_orch, J_new
                )

    return new_orch
