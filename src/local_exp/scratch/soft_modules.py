import jax
import jax.numpy as jnp
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.modules.fully_connected import FullyConnected
from typing import Self
import equinox as eqx


def _prep_xyyhat(x: jax.Array, y: jax.Array, y_hat: jax.Array):
    x = jnp.atleast_2d(x)  # (n, d)
    y = jnp.atleast_2d(y)  # (n, K)
    y_hat = jnp.atleast_2d(y_hat)  # (n, K)
    n, d = x.shape
    if y.shape != y_hat.shape or y.shape[0] != n:
        raise ValueError(
            "y and y_hat must have the same (n, K) shape and share n with x."
        )
    return x, y, y_hat, n, d


def perceptron_rule_sigmoid_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    margin: jax.Array,
    beta: float = 5.0,
) -> jax.Array:
    """
    Soft perceptron update using a sigmoid gate on the margin violation.

    Gate: w = sigmoid(beta * (margin - y*y_hat))
    - For large beta → recovers hard perceptron gating.
    - Returns ΔW (d, K) to ADD to the weights (backward-style).

    Args:
        x: (d,) or (n,d)
        y: (n,K) with entries in {-1,+1}
        y_hat: (n,K) raw scores
        margin: broadcastable to (n,K): scalar, (K,), or (n,K)
        beta: sharpness of the transition around the margin boundary

    Returns:
        ΔW of shape (d, K)
    """
    x, y, y_hat, n, d = _prep_xyyhat(x, y, y_hat)
    margin = jnp.broadcast_to(margin, y.shape)
    # Signed margin
    m = y * y_hat
    # Sigmoid gate (dtype-safe)
    w = jax.nn.sigmoid(beta * (margin - m)).astype(y.dtype)  # (n,K)
    # Weighted Hebbian term (then negated for backward-style)
    update = (x.T @ (w * y)) / (n * d**0.5)  # (d,K)
    return -update


def perceptron_rule_hinge_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    margin: jax.Array,
    normalize: bool = False,
) -> jax.Array:
    """
    Soft perceptron update using a (linear) hinge-like gate.

    Gate (unnormalized): w = relu(margin - y*y_hat)
    If normalize=True:   w = relu(margin - y*y_hat) / (margin + eps)
    - Proportional to violation severity.
    - Returns ΔW (d, K) to ADD to the weights (backward-style).

    Args:
        x: (d,) or (n,d)
        y: (n,K) with entries in {-1,+1}
        y_hat: (n,K) raw scores
        margin: broadcastable to (n,K): scalar, (K,), or (n,K)
        normalize: divide by margin to keep scales comparable across classes

    Returns:
        ΔW of shape (d, K)
    """
    x, y, y_hat, n, d = _prep_xyyhat(x, y, y_hat)
    margin = jnp.broadcast_to(margin, y.shape)
    m = y * y_hat
    # Linear hinge magnitude
    w = jnp.clip(margin - m, a_min=0)  # (n,K)
    if normalize:
        eps = jnp.finfo(w.dtype).eps
        w = w / (margin + eps)
    w = w.astype(y.dtype)
    update = (x.T @ (w * y)) / (n * d**0.5)  # (d,K)
    return -update


class HingeRecurrentDiscrete(RecurrentDiscrete):

    def backward(self, x, y, y_hat):
        dJ = perceptron_rule_hinge_backward(x, y, y_hat, self.threshold)
        dJ = dJ * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.J, zero_update, dJ)
        return new_self


class SoftRecurrentDiscrete(RecurrentDiscrete):

    def backward(self, x, y, y_hat):
        dJ = perceptron_rule_sigmoid_backward(x, y, y_hat, self.threshold)
        dJ = dJ * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.J, zero_update, dJ)
        return new_self


class HingeFullyConnected(FullyConnected):

    def backward(self, x, y, y_hat):
        dW = perceptron_rule_hinge_backward(x, y, y_hat, self.threshold)
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self


class SoftFullyConnected(FullyConnected):

    def backward(self, x, y, y_hat):
        dW = perceptron_rule_sigmoid_backward(x, y, y_hat, self.threshold)
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self
