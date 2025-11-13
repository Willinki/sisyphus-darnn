import operator
from typing import Any, Self
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.interfaces import Layer
from darnax.modules.fully_connected import FullyConnected, FrozenFullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState

from .registry import register_model

KeyArray = Array
PyTree = Any


class StableOutputLayer(Layer):
    """Simple output layer that aggregates predictions via summation.

    The layer leaves activations unchanged and defines a ``reduce`` that
    elementwise-sums a PyTree of predictions. Its backward pass is a no-op
    because it has no trainable parameters.

    Notes
    -----
    The current implementation of :meth:`__call__` returns ``zeros_like(x)``.

    """

    J_D: Array

    def __init__(
        self,
        features: int,
        j_d: ArrayLike,
        key: KeyArray,
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
        self.J_D = j_d_vec

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute the forward pass.

        Parameters
        ----------
        x : Array
            Input tensor; any shape is accepted.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Currently ``zeros_like(x)`` (see Notes in the class docstring).

        """
        return self.J_D * x

    def reduce(self, h: PyTree) -> Array:
        """Elementwise-sum all array leaves in a PyTree of predictions.

        Parameters
        ----------
        h : PyTree
            PyTree whose leaves are arrays with identical shapes and dtypes.

        Returns
        -------
        Array
            The elementwise sum across all leaves.

        Raises
        ------
        ValueError
            If ``h`` has no leaves (as per ``tree_reduce`` semantics).

        Notes
        -----
        Uses :func:`jax.tree_util.tree_reduce` with :data:`operator.add`.

        """
        return jax.numpy.asarray(jax.tree_util.tree_reduce(operator.add, h))

    def activation(self, x: Array) -> Array:
        """Identity activation.

        Parameters
        ----------
        x : Array
            Input tensor.

        Returns
        -------
        Array
            ``x`` unchanged.

        """
        return x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """No-op local update.

        This layer has no trainable parameters, so it returns itself unchanged.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction (unused).

        Returns
        -------
        Self
            ``self`` (no parameter updates).

        """
        return self

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


@register_model("fc-baseline-stable-output")
def build_fc_baseline_stable_output(
    seed: int,
    dim_data: int,
    dim_hidden: int,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
    threshold_j: float,
    threshold_out: float,
    j_d: float,
    j_d_output: float,
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
            2: FullyConnected(
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
            2: StableOutputLayer(features=num_labels, j_d=j_d_output, key=keys[4]),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator
