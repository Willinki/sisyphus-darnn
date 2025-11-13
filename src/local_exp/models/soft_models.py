import jax
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import FrozenFullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from local_exp.scratch.soft_modules import (
    HingeFullyConnected,
    HingeRecurrentDiscrete,
    SoftFullyConnected,
    SoftRecurrentDiscrete,
)

from .registry import register_model


@register_model("fc-baseline-hinge")
def build_fc_baseline_hinge(
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
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Builds the fully connected baseline recurrent model."""
    state = SequentialState((dim_data, dim_hidden, num_labels))

    master_key = jax.random.key(seed)
    keys = jax.random.split(master_key, num=5)

    layer_map = {
        1: {
            0: HingeFullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                key=keys[0],
            ),
            1: HingeRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: FrozenFullyConnected(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=0.0,
                key=keys[2],
            ),
        },
        2: {
            1: HingeFullyConnected(
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


@register_model("fc-baseline-soft")
def build_fc_baseline_hinge(
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
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Builds the fully connected baseline recurrent model."""
    state = SequentialState((dim_data, dim_hidden, num_labels))

    master_key = jax.random.key(seed)
    keys = jax.random.split(master_key, num=5)

    layer_map = {
        1: {
            0: SoftFullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                key=keys[0],
            ),
            1: SoftRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: FrozenFullyConnected(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=0.0,
                key=keys[2],
            ),
        },
        2: {
            1: SoftFullyConnected(
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
