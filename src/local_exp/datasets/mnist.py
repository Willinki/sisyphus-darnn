"""MNIST data utilities with configurable sampling, projections, and transforms.

This module provides a `MNISTData` class that loads the MNIST dataset
(using Hugging Face `datasets`), optionally applies a linear Gaussian
projection, and supports several label encodings and input transforms.
It yields contiguous slices as batches for both training and evaluation.
"""

from typing import Iterator, Literal, Optional
import jax
import jax.numpy as jnp
from datasets import load_dataset
from .registry import register_dataset
import logging

logger = logging.getLogger(__name__)


class Mnist:
    """MNIST dataset with optional linear projection and configurable transforms.

    The class materializes a training subset (optionally balanced per class up to
    a requested cap) and the full test split. A single random projection (if used)
    is shared across both splits for consistency. Training batches are provided
    via contiguous slicing, with precomputed ranges for determinism.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key used for per-class sampling, projection initialization,
        and training shuffle.
    batch_size : int, optional
        Number of examples per batch for iteration. Must be > 1. Default is 64.
    linear_projection : int or None, optional
        Output dimensionality of a Gaussian random projection applied to inputs.
        If ``None``, no projection is applied. Default is 100.
    num_images_per_class : int or None, optional
        If ``None``, the entire training split is used as-is (no per-class capping).
        If an integer ``k``, up to ``k`` examples per class are sampled uniformly
        at random (if a class has fewer than ``k``, all available examples are used).
        Default is ``None``.
    label_mode : {"pm1", "ooe", "c-rescale"}, optional
        Encoding for labels:
        - ``"pm1"``: ±1 one-hot (true class = +1, others = −1).
        - ``"ooe"``: standard 0/1 one-hot.
        - ``"c-rescale"``: +√C/2 at the true class, −0.5 elsewhere (C = #classes).
        Default is ``"c-rescale"``.
    x_transform : {"sign", "tanh", "identity"}, optional
        Nonlinearity applied to inputs after the optional projection:
        - ``"sign"``: elementwise sign with zeros mapped to −1.
        - ``"tanh"``: elementwise hyperbolic tangent.
        - ``"identity"``: no nonlinearity.
        Default is ``"sign"``.

    Attributes
    ----------
    x : jax.Array, shape (N_train, D)
        Training inputs after preprocessing (projection + transform).
    y : jax.Array, shape (N_train, C)
        Encoded training labels.
    x_eval : jax.Array, shape (N_test, D)
        Evaluation (test) inputs after preprocessing (same projection/transform).
    y_eval : jax.Array, shape (N_test, C)
        Encoded evaluation (test) labels.
    input_dim : int
        Final input dimensionality ``D`` (post-projection if used).
    num_batches : int
        Number of training batches given the configured ``batch_size``.
    num_eval_batches : int
        Number of evaluation batches for the test split.
    num_data : int
        Number of preprocessed training examples.
    num_eval_data : int
        Number of preprocessed evaluation (test) examples.

    Notes
    -----
    - A single Gaussian projection matrix is sampled once and shared by both
      train and test. Its variance is scaled by ``1/in_dim`` to keep output
      variance roughly unit per component.
    - Training batches are contiguous slices (no reshuffling between epochs).
      To reshuffle, re-instantiate with a new PRNG key.

    Examples
    --------
    Use the entire training split, 256-D projection, tanh features, standard one-hot labels:

    >>> data = MNISTData(
    ...     key=jax.random.PRNGKey(0),
    ...     batch_size=128,
    ...     linear_projection=256,
    ...     num_images_per_class=None,
    ...     label_mode="ooe",
    ...     x_transform="tanh",
    ... )
    >>> len(data) > 0
    True
    """

    NUM_CLASSES = 10
    FLAT_DIM = 28 * 28

    def __init__(
        self,
        key: jax.Array,
        batch_size: int = 64,
        linear_projection: Optional[int] = 100,
        *,
        num_images_per_class: Optional[int] = None,
        flatten: bool = True,
        label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
        x_transform: Literal["sign", "tanh", "identity"] = "sign",
    ):
        if not (linear_projection is None or isinstance(linear_projection, int)):
            raise TypeError("`linear_projection` must be `None` or `int`.")
        if batch_size <= 1:
            raise ValueError(f"Invalid batch_size={batch_size!r}; must be > 1.")
        if num_images_per_class is not None and num_images_per_class <= 0:
            raise ValueError("`num_images_per_class` must be positive or None.")

        self.linear_projection = linear_projection
        self.flatten = flatten
        self.label_mode = label_mode
        self.x_transform = x_transform
        self.batch_size = int(batch_size)

        # Build arrays once.
        self._create_dataset(key, num_images_per_class)

        # Precompute slicing ranges for train/eval.
        self.num_data = int(self.x.shape[0])
        self.num_batches = -(-self.num_data // self.batch_size)  # ceil div
        self._train_bounds = [
            (i * self.batch_size, min((i + 1) * self.batch_size, self.num_data))
            for i in range(self.num_batches)
        ]
        self.num_eval_data = int(self.x_eval.shape[0])
        self.num_eval_batches = -(-self.num_eval_data // self.batch_size)
        self._eval_bounds = [
            (i * self.batch_size, min((i + 1) * self.batch_size, self.num_eval_data))
            for i in range(self.num_eval_batches)
        ]

    # ------------------------------- Public API ------------------------------- #
    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over training data in contiguous mini-batches.

        Yields
        ------
        (x, y) : tuple of jax.Array
            A tuple containing a batch of inputs ``x`` with shape
            ``(B, D)`` and encoded labels ``y`` with shape ``(B, C)``,
            where ``B`` is ``batch_size``, ``D`` is the processed input
            dimensionality, and ``C`` is the number of classes.

        Notes
        -----
        Batches are contiguous slices computed from precomputed bounds
        for determinism. To introduce epoch-wise shuffling, re-create
        the dataset with a different PRNG key.
        """
        for lo, hi in self._train_bounds:
            yield self.x[lo:hi], self.y[lo:hi]

    def iter_eval(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over evaluation (test) data in contiguous mini-batches.

        Yields
        ------
        (x_eval, y_eval) : tuple of jax.Array
            A tuple containing a batch of evaluation inputs ``x_eval`` with
            shape ``(B, D)`` and encoded labels ``y_eval`` with shape
            ``(B, C)``.

        Notes
        -----
        The evaluation split is not shuffled and uses the same projection/
        transform as the training split.
        """
        for lo, hi in self._eval_bounds:
            yield self.x_eval[lo:hi], self.y_eval[lo:hi]

    def __len__(self) -> int:
        """Return the number of training batches.

        Returns
        -------
        int
            The number of batches yielded by ``__iter__`` given the
            configured ``batch_size``.
        """
        return self.num_batches

    # ------------------------------ Internals ------------------------------ #
    @staticmethod
    def _load_mnist_split(split: str) -> tuple[jax.Array, jax.Array]:
        """Load an MNIST split.

        Parameters
        ----------
        split : {"train", "test"}
            Which split to load.

        Returns
        -------
        x : jax.Array, shape (N, 28, 28), dtype float32
            Images normalized to the range [0, 1].
        y : jax.Array, shape (N,), dtype int32
            Integer class labels in ``[0, 9]``.

        Raises
        ------
        AssertionError
            If ``split`` is not one of ``"train"`` or ``"test"``.
        """
        assert split in ["train", "test"]
        logger.info(f"Loading split {split=}")
        ds = load_dataset("mnist")
        x = jnp.asarray([jnp.array(im) for im in ds[split]["image"]], dtype=jnp.float32)
        y = jnp.asarray(ds[split]["label"], dtype=jnp.int32)
        x = x / 255.0
        logger.info(f"{x.shape=}, {y.shape=}")
        return x, y

    @staticmethod
    def _labels_c_rescale(y_scalar: jax.Array, num_classes: int) -> jax.Array:
        """Encode labels using the c-rescale scheme.

        The true class is set to ``+sqrt(C)/2`` and all others to ``-0.5``,
        where ``C`` is the number of classes.

        Parameters
        ----------
        y_scalar : jax.Array, shape (N,)
            Integer class labels.
        num_classes : int
            Number of classes.

        Returns
        -------
        jax.Array, shape (N, C)
            Encoded labels (float32).
        """
        logger.info("Rescaling labels to C/2, -0.5")
        one_hot = jax.nn.one_hot(y_scalar, num_classes, dtype=jnp.float32)
        return one_hot * (num_classes**0.5 / 2.0) - 0.5

    @staticmethod
    def _labels_pm1(y_scalar: jax.Array, num_classes: int) -> jax.Array:
        """Encode labels as ±1 one-hot.

        Parameters
        ----------
        y_scalar : jax.Array, shape (N,)
            Integer class labels.
        num_classes : int
            Number of classes.

        Returns
        -------
        jax.Array, shape (N, C)
            Encoded labels with +1 at the true class and −1 elsewhere.
        """
        logger.info("Rescaling labels to +-1")
        one_hot = jax.nn.one_hot(y_scalar, num_classes, dtype=jnp.float32)
        return one_hot * 2.0 - 1.0

    @staticmethod
    def _labels_ooe(y_scalar: jax.Array, num_classes: int) -> jax.Array:
        """Encode labels as standard one-hot (0/1).

        Parameters
        ----------
        y_scalar : jax.Array, shape (N,)
            Integer class labels.
        num_classes : int
            Number of classes.

        Returns
        -------
        jax.Array, shape (N, C)
            Encoded labels with 1 at the true class and 0 elsewhere.
        """
        logger.info("Rescaling to one hot encoding")
        return jax.nn.one_hot(y_scalar, num_classes, dtype=jnp.float32)

    @staticmethod
    def _random_projection_matrix(
        key: jax.Array, out_dim: int, in_dim: int
    ) -> jax.Array:
        """Sample a Gaussian random projection matrix.

        Entries are i.i.d. Normal(0, 1/in_dim), yielding approximately unit-variance
        projected features per component.

        Parameters
        ----------
        key : jax.Array
            PRNG key for random sampling.
        out_dim : int
            Number of output dimensions (rows).
        in_dim : int
            Number of input dimensions (columns).

        Returns
        -------
        jax.Array, shape (out_dim, in_dim), dtype float32
            Random projection matrix.
        """
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(
            in_dim
        )

    @staticmethod
    def _take_per_class(
        key: jax.Array, x: jax.Array, y: jax.Array, k_per_class: int
    ) -> tuple[jax.Array, jax.Array]:
        """Uniformly sample up to ``k_per_class`` examples for each class.

        For each class ``c``, all indices are shuffled using a split of the PRNG
        key and the first ``k`` indices are selected, where ``k`` is
        ``min(k_per_class, n_avail_for_c)``.

        Parameters
        ----------
        key : jax.Array
            PRNG key used to shuffle indices per class.
        x : jax.Array, shape (N, D)
            Input features.
        y : jax.Array, shape (N,)
            Integer class labels.
        k_per_class : int
            Maximum number of examples to take per class.

        Returns
        -------
        x_sub : jax.Array, shape (N', D)
            Subset of inputs containing up to ``k_per_class`` examples per class.
        y_sub : jax.Array, shape (N',)
            Corresponding integer labels.

        Notes
        -----
        Classes with fewer than ``k_per_class`` available examples contribute all
        their examples (no error is raised).
        """
        logger.info(f"Taking {k_per_class} elements per class.")
        xs, ys = [], []
        for cls in range(Mnist.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == cls)[0]
            n_avail = int(idx.shape[0])
            k = min(k_per_class, n_avail)
            perm = jax.random.permutation(sub, n_avail)
            take = idx[perm[:k]]
            logger.info(f"For {cls=} took {k}/{n_avail} elements")
            xs.append(x[take])
            ys.append(y[take])
        return jnp.concatenate(xs, axis=0), jnp.concatenate(ys, axis=0)

    def _apply_x_transform(self, x: jax.Array) -> jax.Array:
        """Apply the configured input transform.

        Parameters
        ----------
        x : jax.Array, shape (N, D)
            Input features.

        Returns
        -------
        jax.Array, shape (N, D)
            Transformed features.

        Raises
        ------
        ValueError
            If ``x_transform`` is unknown.
        """
        if self.x_transform == "sign":
            logger.info("Applying sign transform")
            sgn = jnp.sign(x)
            return jnp.where(sgn == 0, jnp.array(-1.0, dtype=sgn.dtype), sgn)
        elif self.x_transform == "tanh":
            logger.info("Applying tanh transform")
            return jnp.tanh(x)
        elif self.x_transform == "identity":
            logger.info("Applying identity transform")
            return x
        else:
            raise ValueError(f"Unknown x_transform={self.x_transform!r}")

    def _encode_labels(self, y_scalar: jax.Array) -> jax.Array:
        """Encode labels according to the configured label mode.

        Parameters
        ----------
        y_scalar : jax.Array, shape (N,)
            Integer class labels.

        Returns
        -------
        jax.Array, shape (N, C)
            Encoded labels.

        Raises
        ------
        ValueError
            If ``label_mode`` is unknown.
        """
        if self.label_mode == "c-rescale":
            return self._labels_c_rescale(y_scalar, self.NUM_CLASSES)
        elif self.label_mode == "ooe":
            return self._labels_ooe(y_scalar, self.NUM_CLASSES)
        elif self.label_mode == "pm1":
            return self._labels_pm1(y_scalar, self.NUM_CLASSES)
        else:
            raise ValueError(f"Unknown label_mode={self.label_mode!r}")

    def _maybe_flatten_and_project(
        self, w: jax.Array | None, x: jax.Array
    ) -> jax.Array:
        """Apply the linear projection if a matrix is provided.

        Parameters
        ----------
        w : jax.Array or None
            Projection matrix of shape ``(out_dim, in_dim)`` or ``None`` to skip.
        x : jax.Array, shape (N, in_dim)
            Input features.

        Returns
        -------
        jax.Array, shape (N, out_dim) or (N, in_dim)
            Projected features if ``w`` is not ``None``, otherwise the input.
        """
        if not self.flatten and w is not None:
            logger.warning(
                "Flatten=False but also projection is not none. Flattening data anyway."
            )
            self.flatten = True
        if self.flatten:
            logger.info("Flattening inputs")
            x = jnp.reshape(x, (x.shape[0], -1))
            logger.info(f"{x.shape=}")
        if w is not None:
            logger.info("Randomly projecting the input")
            x = (x @ w.T).astype(jnp.float32)
        return x

    def _create_dataset(self, key: jax.Array, k_per_class: Optional[int]) -> None:
        """Materialize train subset and test split with consistent preprocessing.

        This method:
          1. Loads raw MNIST train/test splits.
          2. Selects the training subset:
             - If ``k_per_class`` is ``None``, uses the full training split.
             - Else, samples up to ``k_per_class`` examples per class uniformly.
          3. Samples (once) a Gaussian projection and applies it to both splits
             if ``linear_projection`` is not ``None``.
          4. Applies the configured input transform (sign/tanh/identity).
          5. Encodes labels according to the configured label mode.
          6. Shuffles the training split once with the provided PRNG key.

        Parameters
        ----------
        key : jax.Array
            PRNG key for subset sampling, projection, and shuffling.
        k_per_class : int or None
            Per-class cap for training examples, or ``None`` for full train set.

        Returns
        -------
        None
        """
        key_sample, key_proj, key_shuf_tr = jax.random.split(key, 3)

        # Load raw splits.
        x_tr_all, y_tr_all = self._load_mnist_split("train")
        x_ev_all, y_ev_scalar = self._load_mnist_split("test")

        # Train subset selection.
        if k_per_class is None:
            x_tr, y_tr_scalar = x_tr_all, y_tr_all
        else:
            x_tr, y_tr_scalar = self._take_per_class(
                key_sample, x_tr_all, y_tr_all, k_per_class
            )

        # Shared projection across splits.
        w = (
            self._random_projection_matrix(
                key_proj, int(self.linear_projection), (x_tr.shape[-2] * x_tr.shape[-1])
            )
            if self.linear_projection is not None
            else None
        )
        x_tr = self._maybe_flatten_and_project(w, x_tr)
        x_ev = self._maybe_flatten_and_project(w, x_ev_all)

        # Apply chosen x transform.
        x_tr = self._apply_x_transform(x_tr)
        x_ev = self._apply_x_transform(x_ev)

        # Encode labels; shuffle train only.
        y_tr = self._encode_labels(y_tr_scalar)
        perm_tr = jax.random.permutation(key_shuf_tr, x_tr.shape[0])
        self.x, self.y = x_tr[perm_tr], y_tr[perm_tr]

        self.x_eval = x_ev
        self.y_eval = self._encode_labels(y_ev_scalar)

        # Convenience metadata.
        self.input_dim = int(self.x.shape[1])

        # Final logging
        logger.info(f"Final training shape: {self.x.shape}, {self.y.shape}")
        logger.info(f"Final validation shape: {self.x_eval.shape}, {self.y_eval.shape}")


@register_dataset("entangled_mnist")
def build_entangled_mnist(seed: int, batch_size: int, num_images_per_class: int):
    return Mnist(
        jax.random.key(seed),
        batch_size,
        num_images_per_class=num_images_per_class,
        linear_projection=100,
        flatten=True,
        label_mode="c-rescale",
        x_transform="sign",
    )


@register_dataset("general_mnist")
def build_general_mnist(
    seed: int,
    batch_size: int = 64,
    linear_projection: Optional[int] = 100,
    *,
    num_images_per_class: Optional[int] = None,
    flatten: bool = True,
    label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
    x_transform: Literal["sign", "tanh", "identity"] = "sign",
):
    return Mnist(
        jax.random.key(seed),
        batch_size,
        linear_projection,
        num_images_per_class,
        flatten,
        label_mode,
        x_transform,
    )
