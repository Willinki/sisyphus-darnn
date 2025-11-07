import jax
import equinox as eqx
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


def scan_steps(fn, s: SequentialState, rng: jax.Array, steps: int):
    """Scan `steps` times a (s, rng)->(s, rng) transition."""

    def body(carry, _):
        s, rng = carry
        s, rng = fn(s, rng=rng)
        return (s, rng), None

    (s, rng), _ = jax.lax.scan(body, (s, rng), xs=None, length=steps)
    return s, rng


def apply_update(
    orch: SequentialOrchestrator,
    s: SequentialState,
    opt_state,
    rng: jax.Array,
    optimizer,
):
    """Compute local deltas via .backward, then apply Optax updates.

    Why separate this helper?
      - Clear separation of concerns (dynamics vs parameter updates).
      - Easier to unit-test and profile independently.
    """
    grads = orch.backward(s, rng=rng)  # local deltas, tree-shaped like `orch`
    params = eqx.filter(orch, eqx.is_inexact_array)  # trainable leaves
    grads = eqx.filter(grads, eqx.is_inexact_array)  # drop non-arrays from grads

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    orch = eqx.apply_updates(orch, updates)
    return orch, opt_state
