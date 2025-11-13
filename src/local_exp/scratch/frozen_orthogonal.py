import jax
from jax import Array
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from darnax.modules.interfaces import Adapter
from typing import Self
import equinox as eqx

KeyArray = Array


class FrozenRescaledFullyConnected(Adapter):
    """Fully connected trainable adapter with orthogonal init.


    A dense linear projection followed by an elementwise per-output scaling.
    Learning uses a **local perceptron-style rule** parameterized by a
    per-output ``threshold``; only ``W`` receives updates, while
    ``strength`` and ``threshold`` act as (learnable-if-you-want) hyperparameters
    that are **not** updated by :meth:`backward`.

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
    threshold: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
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
        self.threshold = self._set_shape(threshold, out_features, dtype)
        self.W = (
            jax.random.normal(key, (in_features, out_features), dtype=dtype)
            * self.strength
            / jnp.sqrt(in_features)
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
        C = self.W.shape[0]
        a = 1 / 2 * (C / 2 + 1 / 2)
        b = 1 / 2 * (C / 2 - 1 / 2)
        return (x * a + b) @ self.W

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return zero update for all parameters.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction/logits (unused).

        Returns
        -------
        Self
            PyTree of zeros with the same structure as ``self``.

        """
        zero_update: Self = jax.tree.map(jnp.zeros_like, self)
        return zero_update

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
