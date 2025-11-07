#!/usr/bin/env python3
"""
Convert JAX-ready .npz feature files (from `convert_features_to_jax.py`) into a
Hugging Face dataset and push it to the Hub.

This script expects two files (by default placed in `../data/cifar10-features/<variant>`):
- train_set.npz (contains arrays named `x` and `y`)
- eval_set.npz  (contains arrays named `x` and `y`)

Sensible defaults are provided; you can override via CLI flags.

Dependencies:
- datasets
- huggingface_hub
- numpy

Example:
  python scripts/python_scripts/push_jax_npz_to_hf.py \
    --data_dir ../data/cifar10-features --variant mlpl1 \
    --repo_name cifar10-features-mlpl1 --private

If you don't pass a token explicitly, the script will try to use the token
stored by the `huggingface-cli login` command or the `HF_TOKEN` env var.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import logging

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("push_jax_npz_to_hf")


def load_npz_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info("Loading %s", path)
    with np.load(path, allow_pickle=False) as d:
        if "x" not in d or "y" not in d:
            raise ValueError(f"NPZ file {path} must contain arrays named 'x' and 'y'")
        x = d["x"]
        y = d["y"]
    logger.info(
        "Loaded arrays: x.shape=%s, y.shape=%s",
        getattr(x, "shape", None),
        getattr(y, "shape", None),
    )
    return x, y


def to_hf_dataset(x: np.ndarray, y: np.ndarray) -> Dataset:
    """Convert numpy arrays into a Hugging Face Dataset.
    The datasets library can handle numpy arrays directly, no need for conversion to lists.
    """
    # basic validation
    if len(x) != len(y):
        raise ValueError(
            f"Feature and label lengths differ: len(x)={len(x)}, len(y)={len(y)}"
        )
    if len(x.shape) > 2:
        x = x.squeeze()

    # Create dataset directly from numpy arrays
    data = {"x": x, "y": y}
    return Dataset.from_dict(data)


def main(
    data_dir: str = "../data/cifar10-features",
    variant: str = "mlpl1",
    train_file: str = "train_set.npz",
    eval_file: str = "eval_set.npz",
    repo_name: str | None = None,
    namespace: str | None = None,
    token: str | None = None,
    private: bool = False,
    create_repo_if_missing: bool = True,
):
    data_path = Path(data_dir)
    variant_dir = data_path / variant

    if repo_name is None:
        repo_name = f"cifar10-features-{variant}"

    train_path = variant_dir / train_file
    eval_path = variant_dir / eval_file

    train_x, train_y = load_npz_file(train_path)
    eval_x, eval_y = load_npz_file(eval_path)

    train_ds = to_hf_dataset(train_x, train_y)
    eval_ds = to_hf_dataset(eval_x, eval_y)

    dataset_dict = DatasetDict({"train": train_ds, "validation": eval_ds})

    api = HfApi()

    # Determine namespace / username
    if namespace is None:
        try:
            whoami = api.whoami(token=token)
            username = whoami.get("name") or whoami.get("user", {}).get("name")
            if username:
                namespace = username
                logger.info("Detected HF username: %s", username)
        except Exception as e:
            logger.debug("Could not auto-detect HF username: %s", e)

    if namespace is None:
        # If still None, we will try to push using repo_name only; the API requires a full repo id
        logger.warning(
            "No namespace detected. You should pass --namespace or set a token. The script will try to create a repo named '%s' in your account if the token allows it.",
            repo_name,
        )

    # Build full repo id
    full_repo_id = f"{namespace}/{repo_name}" if namespace else repo_name

    # Create dataset repo on the Hub if requested
    if create_repo_if_missing:
        try:
            logger.info(
                "Ensuring dataset repo exists: %s (private=%s)", full_repo_id, private
            )
            api.create_repo(
                name=repo_name,
                token=token,
                repo_type="dataset",
                private=private,
                exist_ok=True,
                organization=namespace if namespace else None,
            )
        except Exception as e:
            logger.warning(
                "Could not create repo (it may already exist or insufficient permissions): %s",
                e,
            )

    # Push DatasetDict to hub
    logger.info("Pushing dataset to Hugging Face Hub as '%s'", full_repo_id)
    try:
        dataset_dict.push_to_hub(repo_id=full_repo_id, token=token)
        logger.info("Successfully pushed dataset to: %s", full_repo_id)
    except Exception as e:
        logger.error("Failed to push dataset to the Hub: %s", e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push train/eval .npz feature files to the Hugging Face Hub as a Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/cifar10-features",
        help="Root data directory that contains variant subfolders",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="mlpl1",
        help="Variant subfolder name where train_set.npz and eval_set.npz live",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train_set.npz",
        help="Train npz filename (inside data_dir/variant)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="eval_set.npz",
        help="Eval npz filename (inside data_dir/variant)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="Repository name for the dataset (e.g. cifar10-features-mlpl1). If omitted it will be inferred from the variant",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Hugging Face username or organization namespace. If omitted the script will attempt to auto-detect from the token",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted the script will try the token stored by 'huggingface-cli login' or HF_TOKEN env var",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private (default: public)",
    )
    parser.add_argument(
        "--no-create-repo",
        dest="create_repo",
        action="store_false",
        help="Do not attempt to create the dataset repo before pushing",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        variant=args.variant,
        train_file=args.train_file,
        eval_file=args.eval_file,
        repo_name=args.repo_name,
        namespace=args.namespace,
        token=args.token,
        private=args.private,
        create_repo_if_missing=args.create_repo,
    )
