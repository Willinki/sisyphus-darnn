from typing import Mapping, Tuple, Any
import equinox as eqx
import jax.tree_util as jtu
import jax.numpy as jnp
import optax

PyTree = Any

"""
Per definire una mappa di ottimizzatori su un modello
Ho definito la seguente funzione. Sostanzialmente prende 
l'orchestratore e un dizionario tipo:
{(i, j): label}
dove (i, j) sono gli indici della module della lmap
e label e' una stringa che decidi tu. Puoi raggruppare 
ogni modulo del tuo modello assegnandogli una label.
Tutti i moduli con la stessa label useranno lo stesso ottimizzatore.
Se un modulo non e' specificato nel dizionario, allora usera' 
la label di default (default_label).
"""


def make_lr_map_v2(
    model,
    overrides: Mapping[Tuple[int, int], str] | None = None,
    default_label: str = "default",
) -> PyTree:
    """
    Given an Equinox model with a field `lmap[i][j]` (matrix of submodules),
    returns a pytree of string labels suitable for optax.multi_transform.

    Args:
        model: orchestrator model instance (must have .lmap).
        overrides: mapping {(i, j): label} for any cells that should get a custom label.
                   Example: {(0, 0): "enc", (0, 1): "dec", (1, 0): "heads"}
        default_label: label name for all other parameters.

    Returns:
        A pytree matching model's parameters, with string labels.
    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def like(tree, value):
        # Broadcast a scalar `value` to a tree with same leaves shape/structure
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    # Start with all parameters labeled as default
    labels = jtu.tree_map(lambda _: default_label, params, is_leaf=eqx.is_array)

    # Apply any (i, j) label overrides
    if overrides:
        for (i, j), label in overrides.items():
            labels = eqx.tree_at(
                lambda m: m.lmap[i][j],
                labels,
                replace=like(params.lmap[i][j], label),
            )

    return labels


"""
Esempio di utilizzo:
    lr_map = make_lr_map_v2(
        orchestrator,
        overrides={(1, 0): "w_in", (1, 1): "j", (2, 1): "w_out", (1, 2): "w_back"},
    )
Dove come vedi ho assegnato un learning rate diverso a W_in, J, W_out, W_back.

Questa mappa di learning rate puo' essere passata ad optax.multi_transform:
"""

optimizer = optax.multi_transform(
    {
        "default": optax.sgd(learning_rate=0.0),
        "w_in": optax.sgd(learning_rate=0.1),
        "w_out": optax.sgd(learning_rate=0.2),
        "w_back": optax.sgd(learning_rate=0.3),
        "j": optax.sgd(learning_rate=0.3),
    },
    lr_map,
)

"""
Ora l'oggetto sopra si comporta esattamente come un ottimizzatore normale di Optax,
ma tratta diversamente ogni modulo.
"""

opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))


"""
Ora, per il weight decay. la soluzione e' mooolto artigianale e provvisoria,
pero' per ora va bene e non da troppo fastidio.
Modificala pure, e' fin troppo semplice, forse puoi farlo addirittura manualemente.
"""


def decay(
    orchestrator: SequentialOrchestrator,
    config: Dict[str, Any],
) -> Any:
    """
    Return a new orchestrator with exponentially decayed weights.

    Each weight matrix is multiplied by (1 - rho), simulating
    weight decay instead of hard normalization.

    Parameters
    ----------
    orchestrator : Any
        The orchestrator module containing submodules in lmap[i][j].
    rho : float
        Decay rate. Each weight is multiplied by (1 - rho).

    Returns
    -------
    Any
        A new orchestrator with decayed weights.
    """
    new_orch = orchestrator

    # W_in
    W_in = jnp.asarray(new_orch.lmap[1][0].W)
    rescaling_win = (
        config["optimizer"]["weight_decay_win"]
        * config["optimizer"]["learning_rate_win"]
        / (config["model"]["kwargs"]["dim_data"] ** 0.5)
    )
    W_in_new = W_in * (1.0 - rescaling_win)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][0].W, new_orch, W_in_new)

    # J
    J = jnp.asarray(new_orch.lmap[1][1].J)
    mask = jnp.asarray(new_orch.lmap[1][1]._mask)  # exclude diagonal from decay
    rescaling_j = (
        config["optimizer"]["weight_decay_j"]
        * config["optimizer"]["learning_rate_j"]
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    J_new = J * (1.0 - rescaling_j * mask)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][1].J, new_orch, J_new)

    # W_out
    W_out = jnp.asarray(new_orch.lmap[2][1].W)
    rescaling_wout = (
        config["optimizer"]["weight_decay_wout"]
        * config["optimizer"]["learning_rate_wout"]
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    W_out_new = W_out * (1.0 - rescaling_wout)
    new_orch = eqx.tree_at(lambda o: o.lmap[2][1].W, new_orch, W_out_new)

    # W_back
    W_out = jnp.asarray(new_orch.lmap[1][2].W)
    rescaling_wout = (
        config["optimizer"]["weight_decay_wback"]
        * config["optimizer"]["learning_rate_wback"]
        / (config["model"]["kwargs"]["num_labels"] ** 0.5)
    )
    W_out_new = W_out * (1.0 - rescaling_wout)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][2].W, new_orch, W_out_new)
    return new_orch


"""
Come vedi vado manualmente a prendere i pesi che voglio riscalare e gli applico una semplice 
trasformazione e ci rimetto dentro manualmente il nuovo peso.

Bisogna stare attenti a non riscalare la diagonale di J, ma per il resto e' fatto.

Il problema del weight decay di optax e' che viene applicato a tutti i parametri di un 
modulo in maniera indistinta, quindi per esempio RecurrentDiscrete.threshold viene riscalato.
Noi vorremmo evitarlo.
Sto cercando una soluzione, ma essendo moduli locali e' probabilmente piu' semplice fare in modo
che ogni modulo gestisca il weight decay internamente, ma per ora va bene cosi'.
"""
