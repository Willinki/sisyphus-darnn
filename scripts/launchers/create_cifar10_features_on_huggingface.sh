#!/usr/bin/env bash
set -euo pipefail

# Simple launcher: call the Python uploader with sensible defaults defined
# inside the script. This launcher is intended to be run from the repository
# root (it will detect the repo root automatically).

# Determine repository root (use git if available, otherwise current working dir)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Defaults (relative to repo root)
DATA_DIR="$REPO_ROOT/data/cifar10-features"
VARIANT="mlpl1"
TRAIN_FILE="train_set.npz"
EVAL_FILE="eval_set.npz"
REPO_NAME="cifar10-features-l"

SCRIPT_PATH="$REPO_ROOT/scripts/python_scripts/push_jax_npz_to_hf.py"

echo "Repo root: $REPO_ROOT"
echo "Using data dir: $DATA_DIR"
echo "Variant: $VARIANT"
echo "Train file: $TRAIN_FILE"
echo "Eval file: $EVAL_FILE"
echo "HF repo name: $REPO_NAME"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: python script not found at $SCRIPT_PATH" >&2
  exit 2
fi

python3 "$SCRIPT_PATH" \
  --data_dir "$DATA_DIR" \
  --variant "$VARIANT" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --repo_name "$REPO_NAME"
