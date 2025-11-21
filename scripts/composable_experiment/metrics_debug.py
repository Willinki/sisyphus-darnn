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
        "W_back": wandb.Histogram(orchestrator.lmap[1][2].W),
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


#
# DEBUG METRIC 6: OVERLAP BETWEEN STATE AND PROTOTYPE for correct
#
def get_overlaps_between_states_and_prototype(batch_id, x, y, orchestrator, state):
    orig_state = state
    y_true = jnp.argmax(y, axis=-1)
    y_pred = jnp.argmax(orchestrator.lmap[2][1](state[1]), axis=-1)
    wrong_mask = y_true == y_pred

    prototype = orchestrator.lmap[1][2](y)
    state_vec = orig_state[-2]
    cosine_distance = jnp.sum(state_vec * prototype, axis=-1) / (
        jnp.linalg.norm(state_vec, axis=-1) * jnp.linalg.norm(prototype, axis=-1) + 1e-8
    )

    return cosine_distance[wrong_mask]


def summarize_cosines(values):
    all_cosines = jnp.concat(values)
    return wandb.Histogram(all_cosines)


#
# DEBUG METRIC 7: OVERLAP BETWEEN STATE AND PROTOTYPE for wrong
#
def get_overlaps_between_states_and_prototype_wrong(
    batch_id, x, y, orchestrator, state
):
    orig_state = state
    y_true = jnp.argmax(y, axis=-1)
    y_pred = jnp.argmax(orchestrator.lmap[2][1](state[1]), axis=-1)
    wrong_mask = y_true != y_pred

    prototype = orchestrator.lmap[1][2](y)
    state_vec = orig_state[-2]
    cosine_distance = jnp.sum(state_vec * prototype, axis=-1) / (
        jnp.linalg.norm(state_vec, axis=-1) * jnp.linalg.norm(prototype, axis=-1) + 1e-8
    )

    return cosine_distance[wrong_mask]


#
# DEBUG METRIC 6: OVERLAP BETWEEN STATE AND PROTOTYPE for correct during training
#
def get_overlaps_between_states_and_prototype_train(
    batch_id, x, y, orchestrator, state
):
    if batch_id % 10 != 0:
        return None
    key = jax.random.key(seed=1234 + batch_id)
    state = state.init(x, y)
    state, key = orchestrator.step(state, key, filter_messages="forward")
    for i in range(5):
        state, key = orchestrator.step(state, key, filter_messages="all")
    s_star = state[-2]
    for i in range(5):
        state, key = orchestrator.step(state, key, filter_messages="forward")
    prototype = orchestrator.lmap[1][2](y)
    s_prime = state[-2]
    cosine_distance = jnp.sum(s_star * prototype, axis=-1) / (
        jnp.linalg.norm(s_star, axis=-1) * jnp.linalg.norm(s_star, axis=-1) + 1e-8
    )
    return cosine_distance


def summarize_cosines_filter(values):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return wandb.Histogram([])
    all_cosines = jnp.concat(filtered)
    return wandb.Histogram(all_cosines)


#
# DEBUG METRIC: overlap between prototype of correct class - biggest overlap of wrong class
#
def get_overlaps_between_states_and_prototype_diff(batch_id, x, y, orchestrator, state):
    if batch_id % 10 != 0:
        return None
    orig_state = state
    y_true = jnp.argmax(y, axis=-1)

    # computing prototypes manually (sic...)
    num_labels = orchestrator.lmap[2][1].W.shape[1]
    labels = jnp.zeros((num_labels, num_labels), dtype=jnp.float32) - 1
    labels = labels.at[jnp.diag_indices(num_labels)].set(1)
    prototypes = orchestrator.lmap[1][2](labels)  # shape (num_labels, dim_hidden)

    # computing overlaps
    state_vec = orig_state[-2]  # (shape (batch_size, dim_hidden))
    sim_matrix = jnp.dot(state_vec, prototypes.T) / (
        jnp.linalg.norm(state_vec, axis=-1) * jnp.linalg.norm(prototypes, axis=-1)
    )  # (shape (batch_size, num_labels))
    true_class_sim = sim_matrix[
        jnp.arange(sim_matrix.shape[0]), y_true
    ]  # shape (batch_size,)
    mask = jax.nn.one_hot(y_true, prototypes.shape[0], dtype=bool)
    wrong_class_sim = jnp.where(mask, -jnp.inf, sim_matrix)
    max_wrong_class_sim = jnp.max(wrong_class_sim, axis=-1)  # shape (batch_size,)
    return true_class_sim - max_wrong_class_sim


def summarize_diff_cosines(values):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return wandb.Histogram([])
    all_diffs = jnp.concat(filtered)
    return wandb.Histogram(all_diffs)


DEBUG_METRICS = {
    "error_class": (misclf_hist_per_batch, misclf_hist_aggregate),
    "overlap_states": (return_internal_states, compute_internal_overlap),
    # "overlap_figures": (return_internal_states, compute_internal_overlap_heatmap),
    "weights": (get_weights, pass_weights),
    "fields": (get_fields, summarize_fields),
    "data": (get_label, summarize_labels),
    "final_state": (get_label, summarize_states),
    # "state_prototype_cosine_correct": (
    #    get_overlaps_between_states_and_prototype,
    #    summarize_cosines,
    # ),
    # "state_prototype_cosine_wrong": (
    #    get_overlaps_between_states_and_prototype_wrong,
    #    summarize_cosines,
    # ),
    # "state_prototype_cosine_training": (
    #    get_overlaps_between_states_and_prototype_train,
    #    summarize_cosines_filter,
    # ),
    "state_overlap_margin": (
        get_overlaps_between_states_and_prototype_diff,
        summarize_diff_cosines,
    ),
}
