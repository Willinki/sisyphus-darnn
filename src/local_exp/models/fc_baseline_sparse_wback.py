import operator
from typing import Any, Self
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.interfaces import Adapter
from darnax.modules.fully_connected import FullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState

from .registry import register_model

KeyArray = Array
PyTree = Any


class SparseFeedback(Adapter):
    """Fully connected trainable adapter ``y = (x @ W) * strength``.

    A sparse linear projection followed by an elementwise per-output scaling.
    Learning is not performed in this adapter. Weights are frozen.
    If in_features = C <= out_features = N:
    Given a vector of size C, it projects the first component on N/C compontents
    of the output, the second component on the second N/C components, and so on.
    If in_features = C > out_features = N, we broadcast the first C/N components of the input
    on the first component of the output, etc...

    Attributes
    ----------
    W : Array
        Weight matrix with shape ``(in_features, out_features)``.
    strength : Array
        Per-output scale, shape ``(out_features,)``; broadcast across the last
        dimension of the forward output.
    threshold : Array
        Per-output margin used by the local update rule, shape ``(out_features,)``.

    Notes
    -----
    - Adapters are *stateless* per the Darnax interface, but they may carry
      trainable parameters. This class advertises trainability through ``W``.
    - The local rule is supplied by
      :func:`darnax.utils.perceptron_rule.perceptron_rule_backward` and is not
      required to be a gradient.

    """

    W: Array
    strength: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize weights and per-output scale/threshold.

        Parameters
        ----------
        in_features : int
            Input dimensionality.
        out_features : int
            Output dimensionality.
        strength : float or ArrayLike
            Scalar (broadcast to ``(out_features,)``) or a vector of
            length ``out_features`` providing the per-output scaling.
        threshold : float or ArrayLike
            Scalar or vector of length ``out_features`` with the per-output
            margins used by the local update rule.
        key : Array
            JAX PRNG key to initialize ``W`` with Gaussian entries scaled by
            ``1/sqrt(in_features)``.
        dtype : DTypeLike, optional
            Dtype for parameters (default: ``jnp.float32``).

        Raises
        ------
        ValueError
            If ``strength`` or ``threshold`` is neither a scalar nor a 1D array
            of the expected length.

        """
        self.strength = self._set_shape(strength, out_features, dtype)
        self.W = jnp.zeros((in_features, out_features), dtype=dtype)
        # defining block structure based on in_out features ratio
        if in_features <= out_features:
            for i in range(in_features):
                start = i * (out_features // in_features)
                end = (i + 1) * (out_features // in_features)
                self.W = self.W.at[i, start:end].set(
                    jnp.ones((out_features // in_features,), dtype=dtype)
                    * self.strength[i]
                )
        else:
            for i in range(out_features):
                start = i * (in_features // out_features)
                end = (i + 1) * (in_features // out_features)
                self.W = self.W.at[start:end, i].set(
                    jnp.ones((in_features // out_features,), dtype=dtype)
                    * self.strength[i]
                )

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute ``y = (x @ W) * strength`` (broadcast on last dim).

        Parameters
        ----------
        x : Array
            Input tensor with trailing dimension ``in_features``. Leading batch
            dimensions (e.g., ``(N, ...)``) are supported via standard matmul
            broadcasting.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Output tensor with trailing dimension ``out_features``.

        """
        return x @ self.W

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a module-shaped local update where only ``ΔW`` is set.

        Parameters
        ----------
        x : Array
            Forward input(s), shape ``(..., in_features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``W`` holds the update ``ΔW`` from the local rule,
            - ``strength`` and ``threshold`` leaves are zeros.

        Notes
        -----
        Calls :func:`darnax.utils.perceptron_rule.perceptron_rule_backward`
        with the stored per-output ``threshold``.

        """
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self)

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Normalize scalar or 1D input to shape ``(dim,)`` and dtype.

        Parameters
        ----------
        x : ArrayLike
            Scalar or 1D array.
        dim : int
            Expected length for a 1D input or broadcasted scalar.
        dtype : DTypeLike
            Target dtype.

        Returns
        -------
        Array
            Vector of shape ``(dim,)`` and dtype ``dtype``.

        Raises
        ------
        ValueError
            If ``x`` is neither scalar nor a 1D array with length ``dim``.

        """
        x = jnp.array(x, dtype=dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != expected {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")


@register_model("fc-baseline-sparse-feedback")
def build_fc_baseline_stable_output(
    seed: int,
    dim_data: int,
    dim_hidden: int,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
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
            1: RecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: SparseFeedback(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                key=keys[2],
            ),
        },
        2: {
            1: SparseFeedback(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                key=keys[3],
            ),
            2: OutputLayer(),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator
