import operator
from typing import Any, Self
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
import equinox as eqx

from darnax.layer_maps.sparse import LayerMap
from darnax.modules.interfaces import Layer
from darnax.modules.fully_connected import (
    FullyConnected,
    FrozenFullyConnected,
    Wback,
    Wout,
)
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.modules.input_output import OutputLayer
from local_exp.scratch.frozen_orthogonal import FrozenRescaledFullyConnected
from darnax.utils.perceptron_rule import perceptron_rule_backward

from .registry import register_model

KeyArray = Array
PyTree = Any


class SparseFullyConnected(FullyConnected):
    _mask: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
        sparsity: float,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        self.strength = self._set_shape(strength, out_features, dtype)
        self.threshold = self._set_shape(threshold, out_features, dtype)
        key_w, key_mask = jax.random.split(key)
        mask = jax.random.bernoulli(
            key_mask, p=1.0 - sparsity, shape=(in_features, out_features)
        )
        W = (
            jax.random.normal(key_w, shape=(in_features, out_features), dtype=dtype)
            * self.strength
            / jnp.sqrt(in_features * (1 - sparsity))
        )
        self._mask = mask
        self.W = W * mask

    def backward(
        self, x: Array, y: Array, y_hat: Array, gate: Array | None = None
    ) -> Self:
        dW = perceptron_rule_backward(x, y, y_hat, self.threshold, gate)
        dW = dW * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self


class SparseRecurrentDiscrete(Layer):
    """Binary (±1) recurrent layer with SPARSE couplings.

    The layer keeps a dense coupling matrix ``J`` (with fixed diagonal
    ``J_D``), a per-unit margin ``threshold`` for local updates, and an
    internal diagonal mask used to zero out self-updates during learning.

    Attributes
    ----------
    J : Array
        Coupling matrix with shape ``(features, features)``.
    J_D : Array
        Diagonal self-couplings, shape ``(features,)``. Mirrors
        ``jnp.diag(J)`` and is kept fixed by masking during updates.
    sparsity: float
        Fraction of zero entries in J.
    threshold : Array
        Per-unit margin used by the local perceptron-style rule,
        shape ``(features,)``.
    strength: float, default=1.0
        Scalar that multiplies all couplings at initialization to increase layer
        influence in the dynamics. Similar to strengths in fully connected
        and ferromagnetic adapters.
    _mask : Array
        Binary matrix (``1 - I``) that zeroes the diagonal of ``ΔJ`` before
        applying updates. Same shape and dtype as ``J``.

    """

    J: Array
    J_D: Array
    threshold: Array
    strength: Array
    _mask: Array

    def __init__(
        self,
        features: int,
        j_d: ArrayLike,
        sparsity: float,
        threshold: ArrayLike,
        key: KeyArray,
        strength: float = 1.0,
        dtype: DTypeLike = jnp.float32,
    ):
        """Construct the layer parameters.

        Initializes a dense coupling matrix ``J`` with i.i.d. Gaussian entries
        scaled by ``1/sqrt(features)`` and sets its diagonal to ``j_d``.
        Stores per-unit margins in ``threshold`` and a diagonal masking matrix
        to keep self-couplings fixed during learning.

        Parameters
        ----------
        features : int
            Number of units (dimension ``d``). Shapes are derived from this.
        j_d : ArrayLike
            Self-couplings (diagonal of ``J``). Either a scalar (broadcast to
            ``(features,)``) or a vector of length ``features``.
        threshold : ArrayLike
            Per-unit margins for the local update rule. Scalar or vector of
            length ``features``.
        key : KeyArray
            JAX PRNG key used to initialize the off-diagonal entries of ``J``.
        strength: float, optional
            Scalar that multiplies all couplings at initialization to increase layer
            influence in the dynamics. Similar to strengths in fully connected
            and ferromagnetic adapters.
        dtype : DTypeLike, optional
            Parameter dtype, by default ``jnp.float32``.

        Raises
        ------
        ValueError
            If ``j_d`` or ``threshold`` is not scalar or a 1D vector with
            length ``features``.

        """
        j_d_vec = self._set_shape(j_d, features, dtype)
        thresh_vec = self._set_shape(threshold, features, dtype)
        strength_vec = jnp.asarray(strength, dtype=dtype)

        diag = jnp.diag_indices(features)
        key_j, key_mask = jax.random.split(key)
        mask = jax.random.bernoulli(
            key_mask, p=1.0 - sparsity, shape=(features, features)
        )
        mask = mask.at[diag].set(0)
        J = (
            jax.random.normal(key_j, shape=(features, features), dtype=dtype)
            / jnp.sqrt(features * (1 - sparsity))
            * strength_vec
        )
        J = J * mask
        J = J.at[diag].set(j_d_vec)

        self.J = J
        self.J_D = j_d_vec
        self.threshold = thresh_vec
        self.strength = strength_vec
        self._mask = mask

    def activation(self, x: Array) -> Array:
        """Hard-sign activation mapping ties to ``+1``.

        Parameters
        ----------
        x : Array
            Pre-activation tensor.

        Returns
        -------
        Array
            ``(+1)`` where ``x >= 0`` and ``(-1)`` otherwise, cast to
            ``x.dtype``.

        Notes
        -----
        This function is separate from :meth:`__call__` so orchestrators can
        decide when to discretize (e.g., training vs inference dynamics).

        """
        return jnp.where(x >= 0, 1, -1).astype(x.dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        r"""Compute pre-activations.

        Performs a dense update:

        .. math::
            h = x \\cdot J

        Parameters
        ----------
        x : Array
            Input state(s). Shape ``(features,)`` or ``(batch, features)``.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Pre-activation tensor with shape ``(features,)`` or
            ``(batch, features)`` matching ``x``.

        """
        return x @ self.J

    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages by summation.

        Parameters
        ----------
        h : PyTree
            PyTree of arrays (e.g., messages from neighbors) to be summed.

        Returns
        -------
        Array
            Elementwise sum over all leaves in ``h``.

        Notes
        -----
        Uses :func:`jax.tree_util.tree_reduce` with :data:`operator.add`.

        """
        return jnp.asarray(jax.tree_util.tree_reduce(operator.add, h))

    def backward(
        self, x: Array, y: Array, y_hat: Array, gate: Array | None = None
    ) -> Self:
        """Compute a module-shaped local update.

        Produces a PyTree of updates where only ``J`` receives a nonzero
        ``ΔJ``; all other fields are zero. The diagonal of ``ΔJ`` is masked
        to zero so self-couplings stay fixed at ``J_D``.

        Parameters
        ----------
        x : Array
            Inputs used to produce the current prediction. Shape
            ``(features,)`` or ``(batch, features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``J`` contains ``ΔJ`` (diagonal zeroed),
            - all other leaves are zeros.

        Notes
        -----
        Calls :func:`darnax.utils.perceptron_rule.perceptron_rule_backward`
        with the stored per-unit ``threshold``. The rule is local and need not
        be a true gradient.

        Examples
        --------
        >>> upd = layer.backward(x, y, y_hat)
        >>> new_params = eqx.tree_at(lambda m: m.J, layer, layer.J + lr * upd.J)

        """
        dJ = perceptron_rule_backward(x, y, y_hat, self.threshold, gate)
        dJ = dJ * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.J, zero_update, dJ)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Normalize a scalar or vector to shape ``(dim,)`` and dtype.

        Parameters
        ----------
        x : ArrayLike
            Scalar or 1D array.
        dim : int
            Expected length for 1D input or broadcasted scalar.
        dtype : DTypeLike
            Target dtype.

        Returns
        -------
        Array
            A vector of shape ``(dim,)`` with dtype ``dtype``.

        Raises
        ------
        ValueError
            If ``x`` is neither scalar nor a 1D array of length ``dim``.

        """
        x = jnp.array(x, dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != features {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")


@register_model("fc-baseline-sparse")
def build_fc_baseline_sparse(
    seed: int,
    dim_data: int,
    dim_hidden: int,
    sparsity: float,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
    threshold_out: float,
    threshold_j: float,
    j_d: float,
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Builds the fully connected baseline recurrent model."""
    state = SequentialState((dim_data, dim_hidden, num_labels))

    master_key = jax.random.key(seed)
    keys = jax.random.split(master_key, num=5)

    layer_map = {
        1: {
            0: FullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                key=keys[0],
            ),
            1: SparseRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                sparsity=sparsity,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: FrozenRescaledFullyConnected(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=0.0,
                key=keys[2],
            ),
        },
        2: {
            1: FullyConnected(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                threshold=threshold_out,
                key=keys[3],
            ),
            2: OutputLayer(),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator


@register_model("fc-baseline-sparse-fully")
def build_fc_baseline_sparse_fully(
    seed: int,
    dim_data: int,
    dim_hidden: int,
    sparsity: float,
    sparsity_win: float,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
    threshold_out: float,
    threshold_back: float,
    threshold_j: float,
    j_d: float,
    use_crossentropy: bool,
    learnable_wback: bool,
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Builds the fully connected baseline recurrent model."""
    state = SequentialState((dim_data, dim_hidden, num_labels))

    master_key = jax.random.key(seed)
    keys = jax.random.split(master_key, num=5)

    layer_map = {
        1: {
            0: SparseFullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                sparsity=sparsity_win,
                key=keys[0],
            ),
            1: SparseRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                sparsity=sparsity,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: Wback(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=threshold_back,
                key=keys[2],
                learnable=learnable_wback,
            ),
        },
        2: {
            1: Wout(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                threshold=threshold_out,
                key=keys[3],
                use_crossentropy=use_crossentropy,
            ),
            2: OutputLayer(),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator
