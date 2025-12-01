#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import sys

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


TORCH_LINE_REGEX = re.compile(r"\[Torch Clf\].*eval_acc=([0-9.]+)")


def parse_eval_acc(stdout: str) -> float:
    """
    Parse the last eval_acc from the Torch classifier logs.

    Looks for lines like:
      [Torch Clf] Epoch 019 | ... | eval_acc=0.9587 | ...
    """
    matches = TORCH_LINE_REGEX.findall(stdout)
    if not matches:
        # Show log to help debugging
        print(
            "Could not find eval_acc in output. Full stdout:\n", stdout, file=sys.stderr
        )
        raise RuntimeError("No eval_acc found in Torch Clf output.")
    return float(matches[-1])


def build_command(trial: optuna.trial.Trial) -> list[str]:
    """
    Build the Hydra command for a single trial.

    This mirrors your original bash command but replaces the tuned
    parameters with Optuna samples.
    """

    # ------------------------
    # Sample hyperparameters
    # ------------------------
    strength_back = trial.suggest_float("strength_back", 1.0, 2.0)
    strength_forth = trial.suggest_float("strength_forth", 4.2, 5.0)
    threshold_in = trial.suggest_float("threshold_in", 1.2, 2.0)
    threshold_j = trial.suggest_float("threshold_j", 1.2, 2.0)
    threshold_back = 0.0
    j_d = trial.suggest_float("j_d", 0.7, 1.0)

    learning_rate_j = trial.suggest_float("learning_rate_j", 0.05, 0.4, log=True)
    learning_rate_win = trial.suggest_float("learning_rate_win", 0.1, 0.4, log=True)

    weight_decay_j = trial.suggest_float("weight_decay_j", 5e-5, 3e-3, log=True)
    weight_decay_win = trial.suggest_float("weight_decay_win", 5e-9, 2e-3, log=True)

    torch_cfl_lr = trial.suggest_float("torch_clf_lr", 0.0001, 0.01, log=True)

    # Optional: give each trial a tag so it's easy to spot in wandb
    trial_tag = f"optuna_trial_{trial.number}"

    # ------------------------
    # Base Hydra command
    # ------------------------
    # If you don't use `uv`, change "uv", "run" to just "python".
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/python_scripts/main.py",
        "--config-name=ours_sparse_fmnist_tuning",
        "model.kwargs.dim_hidden=1000",
        "epochs=20",
        "torch_clf.epochs=20",
        "torch_clf.enabled=true",
        "data.kwargs.x_transform=identity",
        "data.kwargs.linear_projection=null",
        "data.kwargs.batch_size=16",
        "model.name=fc-baseline-sparse-fully",
        "model.kwargs.dim_data=784",
        "model.kwargs.sparsity=0.99",
        "model.kwargs.sparsity_win=0.9",
        "trainer.fake_dynamics.k=0.5",
        f"torch_clf.lr={torch_cfl_lr}",
        f"wandb.run_name={trial_tag}",
        f"wandb.tags=[ours,mnist,tuning,optuna,{trial_tag}]",
    ]

    # ------------------------
    # Hyper-parameter overrides
    # ------------------------
    cmd += [
        f"model.kwargs.strength_back={strength_back}",
        f"model.kwargs.strength_forth={strength_forth}",
        f"model.kwargs.threshold_in={threshold_in}",
        f"model.kwargs.threshold_j={threshold_j}",
        f"model.kwargs.threshold_back={threshold_back}",
        f"model.kwargs.j_d={j_d}",
        f"optimizer.learning_rate_j={learning_rate_j}",
        f"optimizer.learning_rate_win={learning_rate_win}",
        f"optimizer.weight_decay_j={weight_decay_j}",
        f"optimizer.weight_decay_win={weight_decay_win}",
        # keep these from your script
        "optimizer.weight_decay_wout=0.02",
        "optimizer.learning_rate_wout=0.17",
        # if you have learning_rate_wback in config and want to keep it fixed,
        # you can add it here as well.
    ]

    return cmd


# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------


def objective(trial: optuna.trial.Trial) -> float:
    cmd = build_command(trial)

    # Make Hydra print full stacktraces if something goes wrong
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"

    print("=======================================================")
    print("Running trial", trial.number)
    print("Command:", " ".join(cmd))
    print("=======================================================", flush=True)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    stdout = proc.stdout

    # echo logs so you can see them in real time when running sequentially
    print(stdout)

    if proc.returncode != 0:
        # If the training crashes, we prune the trial
        raise optuna.TrialPruned(f"Process exited with code {proc.returncode}")

    eval_acc = parse_eval_acc(stdout)
    # Reference metric: maximize eval accuracy of torch classifier
    return eval_acc


# ---------------------------------------------------------------------
# Main: create study, run optimization
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel trials. >1 uses multi-process parallelism.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_mnist.db",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mnist_sparse_optuna",
    )
    args = parser.parse_args()

    sampler = TPESampler(multivariate=True, group=True)
    # HyperbandPruner ~ ASHA-ish; with our current setup we only get
    # pruning at the trial level (since the metric arrives at the end),
    # but it's still a decent default.
    pruner = HyperbandPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("\n===== Optimization finished =====")
    print("Best trial number:", study.best_trial.number)
    print("Best value (torch_clf/eval_acc):", study.best_value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
