import jax
import jax.numpy as jnp
import wandb


#
# DEBUG METRIC 1: CLASSES OF MISCLASSIFIED EXAMPLES
# logged as histogram
#
def misclf_hist_per_batch(batch_id, x, y, orchestrator, state) -> jax.Array:
    """Return bincount of true classes among misclassified examples in this batch."""
    y_true = jnp.argmax(y, axis=-1)
    y_predicted = jnp.argmax(state[-1], axis=-1)
    wrong_mask = y_true != y_predicted
    return y_true[wrong_mask]


def misclf_hist_aggregate(values):
    """Aggregate per-batch misclassification histograms and return a W&B histogram."""
    if not values:
        return wandb.Histogram([])
    values = jnp.concat(values)
    # Use raw counts directly (W&B handles count-based input)
    return wandb.Histogram(values)


#
# DEBUG METRIC 2: AVERAGE INTRA-CLASS - INTERCLASS overlap
# logged as a number


def return_internal_states(batch_id, x, y, orchestrator, state):
    if batch_id % 10 != 0:
        return None

    return (state[-2], jnp.argmax(y, axis=-1))


def compute_internal_overlap(values):
    """
    values: list of Optional[Tuple[state[B,S], labels[B]]]
      - state is binary {0,1} or bool; labels are int class ids.
    Computes:
      ratio = mean_overlap_same_class / mean_overlap_diff_class
    Overlap uses Ising-style definition:
      Convert {0,1} -> {-1,1} via s = 2*state - 1,
      q(i,j) = (1/S) * sum_k s_i[k] * s_j[k]
    Returns a Python float (loggable in W&B).
    """
    # Filter out None entries
    tuples = [t for t in values if t is not None]
    if len(tuples) == 0:
        return float("nan")

    # Concatenate across batches
    states_list, labels_list = zip(*tuples)
    Xpm = jnp.concatenate([jnp.asarray(s) for s in states_list], axis=0)  # [N, S]
    y = jnp.concatenate([jnp.asarray(l) for l in labels_list], axis=0)  # [N]
    S = Xpm.shape[1]

    # Pairwise overlaps (Gram) q_ij = (1/S) * Xpm_i @ Xpm_j
    gram = (Xpm @ Xpm.T) / float(S)  # [N, N]

    # Build masks for same/different class pairs (upper triangle, no diagonal)
    N = y.shape[0]
    yy = y[:, None]
    same = yy == yy.T
    upper = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    same_mask = same & upper
    diff_mask = (~same) & upper

    # Gather values
    same_vals = gram[same_mask]
    diff_vals = gram[diff_mask]

    # Means with empty-guard
    same_mean = jnp.where(same_vals.size > 0, jnp.mean(same_vals), jnp.nan)
    diff_mean = jnp.where(diff_vals.size > 0, jnp.mean(diff_vals), jnp.nan)

    return {
        "same_class": same_mean,
        "different_class": diff_mean,
        "diff_same_ratio": diff_mean / same_mean,
    }


DEBUG_METRICS = {
    "error_class": (misclf_hist_per_batch, misclf_hist_aggregate),
    "overlap_states": (return_internal_states, compute_internal_overlap),
}
