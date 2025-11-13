import jax.tree_util as jtu
import equinox as eqx

from darnax.orchestrators.interface import AbstractOrchestrator


def symmetrize_w(
    model: AbstractOrchestrator, swap_id_from: tuple[int, int], alpha=0.1
) -> AbstractOrchestrator:
    i, j = swap_id_from
    w_from = model.lmap[i][j].W  # w_out
    w_to = model.lmap[j][i].W  # w_back

    # replace back with out
    scale_factor = (w_from.shape[0] / w_to.shape[0]) ** 0.5
    w_swapped = (w_from.T / scale_factor) * alpha + w_to * (1 - alpha)

    # modify and change
    new_orchestrator = eqx.tree_at(lambda x: x.lmap[j][i].W, model, w_swapped)
    return new_orchestrator
