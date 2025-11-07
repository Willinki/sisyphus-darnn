import jax.numpy as jnp
from darnax.trainers.utils import scan_n
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from jax import Array
import equinox as eqx


@eqx.filter_jit
def compute_overlaps(
    orchestrator: SequentialOrchestrator,
    state: SequentialState,
    x: Array,
    y: Array,
    rng: Array,
) -> list[None | Array]:
    """Compute overlaps between s_prime and s_star for each element in the batch.

    Given x and y, runs a training dynamics and obtains s_prime and s_star with
    given fields. It then computes the overlap. Returns all overlaps.

    Parameters
    ----------
    orchestrator : SequentialOrchestrator
        Given model.
    state : SequentialState
        State of the correct shape
    x : Array
        input batch
    y : Array
        label batch

    Returns
    -------
    Array
        Array of shape (B,) with overlap for each element

    """
    # 1) per-batch init
    state = state.init(x, y)

    # 2) rollout phases
    (state_a, rng), _ = scan_n(
        orchestrator.step,
        (state, rng),
        n_iter=len(state) - 2,
        filter_messages="forward",
    )
    (state_a, rng), _ = scan_n(
        orchestrator.step, (state_a, rng), n_iter=10, filter_messages="all"
    )
    (state_b, rng), _ = scan_n(
        orchestrator.step,
        (state_a, rng),
        n_iter=10,
        filter_messages="forward",
    )

    # compute overlaps for everything except input and output
    overlaps = [None]
    for i in range(1, len(state) - 1):
        overlap = jnp.sum(state_a[i] * state_b[i], axis=-1) / state[i].shape[-1]
        overlaps.append(overlap)
    overlaps.append(None)
    return overlaps
