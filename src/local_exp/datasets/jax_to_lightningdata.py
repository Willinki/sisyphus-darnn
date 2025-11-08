# lightning_datamodule_from_jax.py
from __future__ import annotations

from typing import Callable, Iterator, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl

try:
    import jax
    import jax.numpy as jnp  # noqa: F401  # (only to ensure JAX is available)
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "JAX is required for the JAX->PyTorch Lightning bridge. "
        "Install jax/jaxlib first."
    ) from e


# ---------- utilities ----------


def _jax_to_torch(x: Any) -> torch.Tensor:
    """
    Convert a jax.Array (or nested numpy-compatible) to a torch.Tensor without copying when possible.
    Keeps dtype; you can cast labels later if needed.
    """
    # Most JAX arrays can be viewed as numpy with np.asarray; this transfers to host if needed.
    if hasattr(x, "__array__"):
        arr = np.asarray(x)
        t = torch.from_numpy(arr)
        return t
    # Fallback for python scalars / lists
    return torch.as_tensor(x)


def _maybe_pair_to_torch(batch: Tuple[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    xt = _jax_to_torch(x)
    yt = _jax_to_torch(y)
    return xt, yt


# ---------- iterable dataset wrappers ----------


class _JaxIterableDataset(IterableDataset):
    """
    Thin wrapper that asks for an iterator-producing function and yields
    torch tensors converted from the underlying JAX batches.
    """

    def __init__(
        self,
        make_iter: Callable[[], Iterator[Tuple[Any, Any]]],
        length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._make_iter = make_iter
        self._length = length

    def __iter__(self):
        it = self._make_iter()
        for batch in it:
            yield _maybe_pair_to_torch(batch)

    def __len__(self) -> int:
        # Providing __len__ helps PL show nice progress bars for training.
        if self._length is None:
            raise TypeError("Length is undefined for this split.")
        return self._length


# ---------- main adapter ----------


class LightningFromJaxDataset(pl.LightningDataModule):
    """
    Wraps a `ClassificationDataset` (JAX-side) to a PyTorch Lightning DataModule.

    Assumptions:
    - The JAX dataset yields *already-batched* (x, y) pairs.
    - `__len__` returns the number of *training* batches.
    - `iter_valid()` is optional and may raise NotImplementedError.
    - `iter_test()` is provided.
    - `spec()` returns metadata; we expose some common fields if present.

    Parameters
    ----------
    dataset : ClassificationDataset
        Your dataset instance implementing the abstract interface.
    seed : int
        PRNG seed used for `dataset.build(jax.random.PRNGKey(seed))`.
    num_workers : int
        DataLoader workers. With IterableDataset + external iterator,
        0 is safest unless you are certain your iterator is worker-safe.
    pin_memory : bool
        Whether to pin memory in DataLoaders.
    persistent_workers : bool
        Propagate to DataLoaders.
    """

    def __init__(
        self,
        dataset,
        *,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.ds = dataset
        self.seed = seed

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        # memoized flags/values
        self._built = False
        self._has_valid = None  # determined in setup
        self._train_len = None

        # surface some spec metadata if available
        try:
            s = self.ds.spec()
            self.num_classes: Optional[int] = s.get("num_classes")
            self.input_shape: Optional[Tuple[int, ...]] = (
                tuple(s["input_shape"]) if "input_shape" in s else None
            )
            self.metadata = s
        except Exception:
            self.num_classes = None
            self.input_shape = None
            self.metadata = {}

    # ---- Lightning API ----

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self._built:
            key = jax.random.PRNGKey(self.seed)
            self.ds.build(key)
            self._built = True

            # detect if validation exists
            try:
                _ = self.ds.iter_valid()
                # If it didn't raise, test is available; but we must not *consume* it here.
                ## ATTENTION: Validation and test set are inverted
                self._has_test = True
            except NotImplementedError:
                self._has_test = False

            # record training length (number of *batches*)
            try:
                self._train_len = int(len(self.ds))
            except Exception:
                self._train_len = None

    # Split-specific iterator factories (fresh iterator every epoch)
    def _make_train_iter(self):
        return iter(self.ds)

    def _make_valid_iter(self):
        # Will only be called if _has_valid is True.
        return self.ds.iter_valid()

    def _make_test_iter(self):
        return self.ds.iter_test()

    # ---- DataLoaders ----

    def train_dataloader(self) -> DataLoader:
        ds = _JaxIterableDataset(self._make_train_iter, length=self._train_len)
        return DataLoader(
            ds,
            batch_size=None,  # batches are prebatched by JAX dataset
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        # test and validation set are inverted
        if not self._has_test:
            return None
        ds = _JaxIterableDataset(self._make_valid_iter, length=None)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        # test and validation set are inverted
        ds = _JaxIterableDataset(self._make_test_iter, length=None)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


# ---------- convenience factory ----------


def make_lightning_datamodule_from_jax(
    dataset,
    *,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> LightningFromJaxDataset:
    """
    Convenience function returning a ready-to-use Lightning DataModule.
    """
    return LightningFromJaxDataset(
        dataset,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


# ---------- example (commented) ----------
# from your_module import MyJaxDataset
# jax_ds = MyJaxDataset(batch_size=128, ...)
# dm = make_lightning_datamodule_from_jax(jax_ds, seed=42, num_workers=0)
# trainer = pl.Trainer(max_epochs=10, ...)
# trainer.fit(model, datamodule=dm)
