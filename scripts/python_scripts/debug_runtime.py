# debug_runtime.py
# debug_runtime.py
from typing import Dict, Tuple, List
import jax

JaxArray = jax.Array


def init_debug_buckets(debug_metrics: Dict[str, Tuple[callable, callable]]):
    return {name: [] for name in debug_metrics.keys()}


def update_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
    batch_id: int,
    x: jax.Array,
    y: jax.Array,
    orchestrator: object,
    state: object,
):
    for name, (per_batch, _) in debug_metrics.items():
        buckets[name].append(per_batch(batch_id, x, y, orchestrator, state))


def aggregate_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
):
    aggregated = {}
    for name, (_, aggregate) in debug_metrics.items():
        aggregated[name] = aggregate(buckets[name])
    return aggregated


def flatten_for_logging(
    prefix: str, aggregated: Dict[str, object]
) -> Dict[str, object]:
    out = {}
    for k, v in aggregated.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{prefix}{k}/{kk}"] = vv
        else:
            out[f"{prefix}{k}"] = v
    return out
