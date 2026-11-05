"""Hyperparameter tuning (random search with an inner validation split).

Protocol
--------
* Tuning data = the TRAINING half of repetition 0, fold 0 of the 5x2 CV. The
  test halves are never touched while tuning.
* That training half is split again (80/20, stratified for classification) into
  an inner-train and a validation part, N_INNER times. Features are
  standardized on the inner-train part only.
* Each candidate configuration is trained for up to MAX_EPOCHS. The validation
  loss curves are averaged over the inner splits; the epoch with the lowest
  averaged loss is the tuned number of epochs, and that loss is the config's score.
* The configuration with the lowest score wins. The winners are saved to
  results/best_hyperparams.json and then used for the full 5x2 runs.

Fair comparison
---------------
ff2, ae_finetune and ae_frozen are given the SAME number of trials and the
SAME draws of learning rate, momentum, activation and layer-width index, so
the only thing that differs is the architecture family:
    ff2          hidden = (h1, h2)           h1 from the width candidates
    ae_*         encoder e = h1 candidate    (e < d), hidden2 = h2
Width candidates are ceil(d/4), ceil(d/2), ceil(3d/4). The autoencoder's own
pretraining settings (ae_lr, ae_momentum, ae_epochs) are fixed, not tuned.
The linear model has no hidden layers, so it gets the full lr x momentum grid.

    python tuning.py --datasets breast_cancer hardware --trials 30
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import autoEncoder
import crossValidation as cv
import experiment as ex
import preProcess as pp

RESULTS_DIR = Path(__file__).resolve().parent / "results"

LR_GRID = [0.003, 0.01, 0.03]
MOMENTUM_GRID = [0.0, 0.9]
ACTIVATION_GRID = ["tanh", "sigmoid"]
H2_GRID = [4, 8, 16]
WIDTH_FRACTIONS = (0.25, 0.5, 0.75)

MAX_EPOCHS = 100
BATCH_SIZE = 32
N_TRIALS = 30
N_INNER = 2
VAL_FRACTION = 0.2
TUNABLE_MODELS = ("linear", "ff2", "ae_finetune", "ae_frozen")


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
def width_candidates(d):
    """Candidate widths for the first hidden layer / the encoder."""
    return sorted({max(1, math.ceil(d * f)) for f in WIDTH_FRACTIONS})


def sample_configs(model, d, n_trials=N_TRIALS, seed=cv.DEFAULT_SEED):
    """Candidate configurations for a model (without epochs/batch size)."""
    if model not in TUNABLE_MODELS:
        raise KeyError(f"Unknown model '{model}'. Choose from {list(TUNABLE_MODELS)}")
    if model == "linear":
        return [{"lr": lr, "momentum": m} for lr, m in itertools.product(LR_GRID, MOMENTUM_GRID)]

    widths = width_candidates(d)
    if model != "ff2" and max(widths) >= d:
        raise ValueError(f"d={d} is too small for an autoencoder grid (needs e < d)")
    combos = list(itertools.product(LR_GRID, MOMENTUM_GRID, ACTIVATION_GRID,
                                    range(len(widths)), range(len(H2_GRID))))
    rng = np.random.default_rng(seed)               # same seed -> same picks for every model
    picks = rng.choice(len(combos), size=min(n_trials, len(combos)), replace=False)

    configs = []
    for p in picks:
        lr, mom, activation, wi, hi = combos[int(p)]
        c = {"lr": lr, "momentum": mom, "activation": activation}
        if model == "ff2":
            c["hidden"] = (widths[wi], H2_GRID[hi])
        else:
            c["encoder"], c["hidden2"] = widths[wi], H2_GRID[hi]
        configs.append(c)
    return configs


def full_hyperparams(model, config, epochs, batch_size=BATCH_SIZE):
    """Complete hyperparameter dict in the format experiment.py expects."""
    base = dict(autoEncoder.DEFAULT_AE_HYPERPARAMS) if model.startswith("ae_") else {}
    return {**base, **config, "epochs": int(epochs), "batch_size": batch_size}


# --------------------------------------------------------------------------
# Validation splits (inside the training half)
# --------------------------------------------------------------------------
def tuning_train_indices(y, task, seed=cv.DEFAULT_SEED):
    """Rows available for tuning: the training half of repetition 0, fold 0."""
    return cv.five_by_two_splits(y, task, seed=seed)[0][0]


def inner_splits(y, task, n_inner=N_INNER, val_fraction=VAL_FRACTION, seed=cv.DEFAULT_SEED):
    """n_inner random (train, validation) splits of positions 0..len(y)-1."""
    y = np.asarray(y)
    n = len(y)
    out = []
    for j in range(n_inner):
        rng = np.random.default_rng(seed + 1000 + j)
        groups = [np.where(y == c)[0] for c in np.unique(y)] if task == "classification" \
            else [np.arange(n)]
        val = []
        for idx in groups:
            idx = rng.permutation(idx)
            n_val = int(round(val_fraction * len(idx)))
            n_val = min(max(n_val, 1), len(idx) - 1) if len(idx) >= 2 else 0
            val.extend(idx[:n_val])
        val = np.sort(np.array(val, dtype=int))
        out.append((np.setdiff1d(np.arange(n), val), val))
    return out


# --------------------------------------------------------------------------
# Scoring one configuration
# --------------------------------------------------------------------------
def evaluate_config(model, hp, prepared, task, k, max_epochs=MAX_EPOCHS, seed=cv.DEFAULT_SEED):
    """Train on each inner split; return (best_epoch, best_validation_loss).

    `prepared` is a list of (Xtr, Xval, ytr, yval), already standardized.
    A diverged run scores infinity instead of crashing the search.
    """
    n_out = k if task == "classification" else 1
    curves = []
    with np.errstate(all="ignore"):
        for j, (Xtr, Xva, ytr, yva) in enumerate(prepared):
            net = ex.MODEL_BUILDERS[model](Xtr.shape[1], n_out, task, hp, seed + j, Xtr=Xtr)
            hist = net.fit(Xtr, ytr, epochs=max_epochs, lr=hp["lr"], batch_size=hp["batch_size"],
                           momentum=hp["momentum"], X_val=Xva, Y_val=yva)
            curves.append(hist["val_loss"])
        mean_curve = np.mean(curves, axis=0)
    mean_curve = np.where(np.isfinite(mean_curve), mean_curve, np.inf)
    best = int(np.argmin(mean_curve))
    return best + 1, float(mean_curve[best])


# --------------------------------------------------------------------------
# Tuning a dataset
# --------------------------------------------------------------------------
def tune_dataset(name, models=TUNABLE_MODELS, n_trials=N_TRIALS, n_inner=N_INNER,
                 max_epochs=MAX_EPOCHS, seed=cv.DEFAULT_SEED, verbose=True):
    """Tune every model on one dataset. Returns (best_hyperparams, log DataFrame)."""
    X, y, task, k = pp.load_dataset(name)
    train_idx = tuning_train_indices(y, task, seed)
    inner = inner_splits(y[train_idx], task, n_inner, seed=seed)
    prepared = []
    for tr_l, va_l in inner:
        tr, va = train_idx[tr_l], train_idx[va_l]
        prepared.append(pp.preprocess_fold(X[tr], X[va], y[tr], y[va], task, k)[:4])
    d = X.shape[1]

    best, rows = {}, []
    for model in models:
        top = None
        for t, config in enumerate(sample_configs(model, d, n_trials, seed)):
            hp = full_hyperparams(model, config, max_epochs)
            start = time.time()
            epoch, score = evaluate_config(model, hp, prepared, task, k, max_epochs, seed)
            rows.append({"dataset": name, "model": model, "trial": t, **config,
                         "best_epoch": epoch, "val_loss": score,
                         "seconds": time.time() - start})
            if top is None or score < top[0]:
                top = (score, full_hyperparams(model, config, epoch))
        best[model] = top[1]
        if verbose:
            print(f"{name:14s} {model:12s} best val loss {top[0]:.4f}  "
                  f"({len(sample_configs(model, d, n_trials, seed))} trials)")
    return best, pd.DataFrame(rows)


def run_tuning(datasets=None, models=TUNABLE_MODELS, out_dir=RESULTS_DIR, **kwargs):
    datasets = list(pp.DATASETS) if datasets is None else list(datasets)
    all_best, logs = {}, []
    for name in datasets:
        best, log = tune_dataset(name, models, **kwargs)
        all_best[name] = best
        logs.append(log)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = pd.concat(logs, ignore_index=True)
    log.to_csv(out_dir / "tuning_log.csv", index=False)
    with open(out_dir / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(all_best, f, indent=2)
    return all_best, log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(pp.DATASETS))
    parser.add_argument("--models", nargs="+", default=list(TUNABLE_MODELS), choices=list(TUNABLE_MODELS))
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--inner", type=int, default=N_INNER)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    a = parser.parse_args()
    run_tuning(a.datasets, a.models, a.out_dir, n_trials=a.trials, n_inner=a.inner,
               max_epochs=a.max_epochs)
    print(f"\nSaved tuning_log.csv and best_hyperparams.json to {a.out_dir}")
