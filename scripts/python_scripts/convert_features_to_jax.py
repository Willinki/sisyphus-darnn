import os
from pathlib import Path
import torch
import jax
import numpy as np
import jax.numpy as jnp


def open_data_inmemory(
    variant: str, data_path: Path, split: str
) -> tuple[torch.Tensor]:
    dir = data_path / f"{variant}"
    tensors_x = []
    tensors_y = []
    files = list(os.listdir(dir))
    files_ids = set(int(x.split("_")[-1].split(".")[0]) for x in files if split in x)
    for file_id in files_ids:
        x_name = f"{split}_{variant}_{file_id}.pt"
        y_name = f"{split}_y_{file_id}.pt"
        tensors_x.append(torch.load(dir / x_name))
        tensors_y.append(torch.load(dir / y_name))
    return torch.cat(tensors_x), torch.cat(tensors_y)


def open_rescaled_data(variant: str, data_path: Path):
    train_x, train_y = open_data_inmemory(variant, data_path, "train")
    train_mean, train_std = train_x.mean(), train_x.std()
    eval_x, eval_y = open_data_inmemory(variant, data_path, "eval")
    train_x_rescaled = (train_x - train_mean) / train_std
    eval_x_rescaled = (eval_x - train_mean) / train_std
    return (train_x_rescaled, train_y), (eval_x_rescaled, eval_y)


def torch_to_jax(t: torch.Tensor):
    return jnp.array(t.detach().cpu().numpy())


def main(
    data_dir: str = "../data/cifar10-features",
    variant: str = "mlpl1",
    output_dir: str = None,
):
    data_path = Path(data_dir)
    if output_dir is None:
        out_dir = data_path / variant
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (train_x_rescaled, train_y), (eval_x_rescaled, eval_y) = open_rescaled_data(
        variant, data_path
    )

    train_x_jax = torch_to_jax(train_x_rescaled)
    train_y_jax = torch_to_jax(train_y).astype(jnp.int32)
    eval_x_jax = torch_to_jax(eval_x_rescaled)
    eval_y_jax = torch_to_jax(eval_y).astype(jnp.int32)

    np.savez_compressed(
        out_dir / "train_set.npz",
        x=jax.device_get(train_x_jax),
        y=jax.device_get(train_y_jax),
    )
    np.savez_compressed(
        out_dir / "eval_set.npz",
        x=jax.device_get(eval_x_jax),
        y=jax.device_get(eval_y_jax),
    )

    print(f"Saved JAX datasets to: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PyTorch CIFAR10 features to JAX .npz format."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cifar10-features",
        help="Directory containing feature folders",
    )
    parser.add_argument(
        "--variant", type=str, default="mlpl1", help="Feature variant subfolder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to data_dir/variant)",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, variant=args.variant, output_dir=args.output_dir)
