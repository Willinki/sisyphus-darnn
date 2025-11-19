import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
import numpy as np


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
# DEBUG METRIC 2: AVERAGE INTRA-CLASS - INTERCLASS overlap also as heatmap
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


def compute_internal_overlap_heatmap(values):
    """
    values: list of Optional[Tuple[state[B,S], labels[B]]]
      - state is already in {-1,+1}; labels are int class ids.

    Produces a heatmap of pairwise overlaps q(i,j) = (1/S) * sum_k s_i[k] * s_j[k],
    grouping samples by class to show block structure.
    """
    tuples = [t for t in values if t is not None]
    if not tuples:
        return None

    # concatenate
    states_list, labels_list = zip(*tuples)
    Xpm = jnp.concatenate(states_list, axis=0)  # [N, S]
    y = jnp.concatenate(labels_list, axis=0)  # [N]

    # compute pairwise overlaps
    S = Xpm.shape[1]
    gram = (Xpm @ Xpm.T) / float(S)  # [N,N]

    # reorder by class
    classes = np.unique(np.array(y))
    order = np.concatenate([np.where(np.array(y) == c)[0] for c in classes])
    gram_ord = gram[order][:, order]
    y_ord = np.array(y)[order]

    # compute block boundaries
    boundaries = []
    start = 0
    for c in classes:
        n = np.sum(y_ord == c)
        boundaries.append((start, start + n))
        start += n

    # plot heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        np.array(gram_ord), vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest"
    )

    for s, e in boundaries:
        if s > 0:
            ax.axhline(s - 0.5, color="k", lw=1)
            ax.axvline(s - 0.5, color="k", lw=1)

    ax.set_title("Internal State Overlaps (grouped by class)")
    ax.set_xlabel("samples (grouped by class)")
    ax.set_ylabel("samples (grouped by class)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="q(i,j)")
    return fig


#
# DEBUG METRIC 3: weights
# logged as a histogram
def get_weights(batch_id, x, y, orchestrator, state):
    if batch_id != 0:
        return None

    return {
        "J": wandb.Histogram(orchestrator.lmap[1][1].J),
        "W_in": wandb.Histogram(orchestrator.lmap[1][0].W),
        "W_out": wandb.Histogram(orchestrator.lmap[2][1].W),
    }


def pass_weights(values):
    values = [x for x in values if x is not None]
    assert len(values) == 1
    return values[0]


#
# DEBUG METRIC 4: INTERNAL FIELDS
#
def get_fields(batch_id, x, y, orchestrator, state):
    # no effect
    state = state.replace_val(-1, y)
    sub = jax.random.key(seed=44)
    left_field = orchestrator.lmap[1][0](state[0])
    right_field = orchestrator.lmap[1][2](state[2])
    self_field = orchestrator.lmap[1][1](state[1])
    return (left_field, self_field, right_field)


def summarize_fields(values):
    left_fields = jnp.concat(list(map(lambda x: x[0], values)))
    self_fields = jnp.concat(list(map(lambda x: x[1], values)))
    right_fields = jnp.concat(list(map(lambda x: x[2], values)))
    return {
        "left_fields": wandb.Histogram(left_fields),
        "right_fields": wandb.Histogram(right_fields),
        "self_fields": wandb.Histogram(self_fields),
    }


#
# DEBUG METRIC 4: LABEL FIELDS
#
def get_label(batch_id, x, y, orchestrator, state):
    return (x.flatten(), y.flatten())


def summarize_labels(values):
    x = jnp.concat(list(map(lambda x: x[0], values)))
    y = jnp.concat(list(map(lambda x: x[1], values)))
    return {
        "x": wandb.Histogram(x),
        "y": wandb.Histogram(y),
    }


#
# DEBUG METRIC 5: LABEL FIELDS
#
def get_label(batch_id, x, y, orchestrator, state):
    return state[-2]


def summarize_states(values):
    state = jnp.concat(values)
    return state.mean()


DEBUG_METRICS = {
    "error_class": (misclf_hist_per_batch, misclf_hist_aggregate),
    "overlap_states": (return_internal_states, compute_internal_overlap),
    # "overlap_figures": (return_internal_states, compute_internal_overlap_heatmap),
    "weights": (get_weights, pass_weights),
    "fields": (get_fields, summarize_fields),
    "data": (get_label, summarize_labels),
    "final_state": (get_label, summarize_states),
}
