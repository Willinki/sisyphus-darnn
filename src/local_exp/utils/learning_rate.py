import jax.tree_util as jtu
import equinox as eqx
from typing import Mapping, Tuple

from darnax.orchestrators.interface import AbstractOrchestrator
from darnax.utils.typing import PyTree


def make_lr_map(
    model: AbstractOrchestrator,
    special_idx: tuple[int, int],
    default_label="default",
    special_label="special",
) -> PyTree:
    """
    Given an Equinox model with a field `lmap[i][j]` (matrix of submodules),
    returns a pytree of labels suitable for optax.multi_transform.

    Args:
        model: orchestrator model instance (must have .lmap).
        special_idx: tuple (i, j) of the submodule that should get a different LR.
        default_label: label name for all other parameters.
        special_label: label name for the special submodule.

    Returns:
        A pytree matching model's parameters, with string labels.

    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def like(tree, value):
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    i, j = special_idx

    # Start with all parameters labeled as default
    labels = jtu.tree_map(lambda _: default_label, params, is_leaf=eqx.is_array)

    # Replace the special cell with its special label
    labels = eqx.tree_at(
        lambda m: m.lmap[i][j],
        labels,
        replace=like(params.lmap[i][j], special_label),
    )

    return labels


def make_lr_map_v2(
    model,
    overrides: Mapping[Tuple[int, int], str] | None = None,
    default_label: str = "default",
) -> "PyTree":
    """
    Given an Equinox model with a field `lmap[i][j]` (matrix of submodules),
    returns a pytree of string labels suitable for optax.multi_transform.

    Args:
        model: orchestrator model instance (must have .lmap).
        overrides: mapping {(i, j): label} for any cells that should get a custom label.
                   Example: {(0, 0): "enc", (0, 1): "dec", (1, 0): "heads"}
        default_label: label name for all other parameters.

    Returns:
        A pytree matching model's parameters, with string labels.
    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def like(tree, value):
        # Broadcast a scalar `value` to a tree with same leaves shape/structure
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    # Start with all parameters labeled as default
    labels = jtu.tree_map(lambda _: default_label, params, is_leaf=eqx.is_array)

    # Apply any (i, j) label overrides
    if overrides:
        for (i, j), label in overrides.items():
            labels = eqx.tree_at(
                lambda m: m.lmap[i][j],
                labels,
                replace=like(params.lmap[i][j], label),
            )

    return labels
