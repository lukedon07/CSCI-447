"""Run models on datasets with 5x2 cross-validation and log every run.

One row per (dataset, model, repetition, fold) is returned and written to CSV.
Available models:
    linear       : inputs -> output  (logistic regression / linear regression)
    ff2          : inputs -> hidden 1 -> hidden 2 -> output
    ae_finetune  : autoencoder-pretrained encoder, then every layer is trained
    ae_frozen    : autoencoder-pretrained encoder kept locked, other layers trained

The hyperparameters below are placeholders; pass --hyperparams
results/best_hyperparams.json (written by tuning.py) to use tuned values.

    python experiment.py --datasets breast_cancer hardware --models linear ff2
    python experiment.py --models linear ff2 ae_finetune ae_frozen --hyperparams results/best_hyperparams.json
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import autoEncoder
import crossValidation as cv
import metrics
import preProcess as pp
from network import FeedForwardNetwork

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_HYPERPARAMS = {
    "linear": {"lr": 0.01, "momentum": 0.9, "epochs": 100, "batch_size": 32},
    "ff2": {"hidden": (16, 8), "activation": "tanh", "lr": 0.01,
            "momentum": 0.9, "epochs": 100, "batch_size": 32},
    "ae_finetune": {**autoEncoder.DEFAULT_AE_HYPERPARAMS,
                    "lr": 0.01, "momentum": 0.9, "epochs": 100, "batch_size": 32},
    "ae_frozen": {**autoEncoder.DEFAULT_AE_HYPERPARAMS,
                  "lr": 0.01, "momentum": 0.9, "epochs": 100, "batch_size": 32},
}


def build_linear(n_in, n_out, task, hp, seed, Xtr=None):
    output = "softmax" if task == "classification" else "linear"
    return FeedForwardNetwork([n_in, n_out], "sigmoid", output, seed=seed)


def build_ff2(n_in, n_out, task, hp, seed, Xtr=None):
    output = "softmax" if task == "classification" else "linear"
    return FeedForwardNetwork([n_in, *hp["hidden"], n_out], hp["activation"], output, seed=seed)


def build_ae_finetune(n_in, n_out, task, hp, seed, Xtr=None):
    return autoEncoder.build_autoencoder_network(Xtr, n_out, task, hp, seed, freeze_encoder=False)


def build_ae_frozen(n_in, n_out, task, hp, seed, Xtr=None):
    return autoEncoder.build_autoencoder_network(Xtr, n_out, task, hp, seed, freeze_encoder=True)


MODEL_BUILDERS = {"linear": build_linear, "ff2": build_ff2,
                  "ae_finetune": build_ae_finetune, "ae_frozen": build_ae_frozen}


def _hp_for(hyperparams, dataset, model):
    """Pick hyperparameters: per-dataset tuned values, else a per-model override,
    else the defaults. `hyperparams` is {dataset: {model: hp}} and/or {model: hp}."""
    if model in hyperparams.get(dataset, {}):
        return hyperparams[dataset][model]
    if model in hyperparams:
        return hyperparams[model]
    return DEFAULT_HYPERPARAMS[model]


def load_hyperparams(path):
    """Read the JSON written by tuning.py -> {dataset: {model: hp}}."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for models in data.values():
        for hp in models.values():
            if "hidden" in hp:
                hp["hidden"] = tuple(hp["hidden"])
    return data


def run_fold(dataset, model, X, y, task, k, train_idx, test_idx, hp, seed):
    """Train one model on one fold and return a dict of results."""
    Xtr, Xte, ytr, yte, y_stats = pp.preprocess_fold(
        X[train_idx], X[test_idx], y[train_idx], y[test_idx], task, k)
    n_out = k if task == "classification" else 1
    net = MODEL_BUILDERS[model](Xtr.shape[1], n_out, task, hp, seed, Xtr=Xtr)

    start = time.time()
    history = net.fit(Xtr, ytr, epochs=hp["epochs"], lr=hp["lr"],
                      batch_size=hp["batch_size"], momentum=hp["momentum"])
    seconds = time.time() - start

    if task == "classification":
        result = metrics.classification_metrics(y[test_idx], net.predict_labels(Xte), k)
    else:
        mean, std = y_stats
        result = metrics.regression_metrics(y[test_idx], net.predict(Xte) * std + mean)

    ae = getattr(net, "autoencoder", None)       # reconstruction error (autoencoder models only)
    recon_train = ae.loss(Xtr, Xtr) if ae is not None else np.nan
    recon_test = ae.loss(Xte, Xte) if ae is not None else np.nan

    return {"dataset": dataset, "task": task, "model": model,
            "n_train": len(train_idx), "n_test": len(test_idx),
            **result,
            "final_train_loss": history["train_loss"][-1],
            "recon_train": recon_train, "recon_test": recon_test,
            "seconds": seconds, "hyperparams": str(hp)}


def run_experiments(datasets=None, models=("linear", "ff2"), hyperparams=None,
                    seed=cv.DEFAULT_SEED, verbose=True):
    """Run every (dataset, model) over all 10 folds. Returns a DataFrame."""
    datasets = list(pp.DATASETS) if datasets is None else list(datasets)
    hyperparams = hyperparams or {}
    for m in models:
        if m not in MODEL_BUILDERS:
            raise KeyError(f"Unknown model '{m}'. Choose from {list(MODEL_BUILDERS)}")

    rows = []
    for name in datasets:
        X, y, task, k = pp.load_dataset(name)
        splits = cv.five_by_two_splits(y, task, seed=seed)      # identical for every model
        for model in models:
            for i, (tr, te) in enumerate(splits):
                rep, fold = divmod(i, 2)
                row = run_fold(name, model, X, y, task, k, tr, te,
                               _hp_for(hyperparams, name, model), seed + i)
                rows.append({**row, "rep": rep, "fold": fold})
            if verbose:
                part = [r for r in rows if r["dataset"] == name and r["model"] == model]
                err = np.mean([r["error"] for r in part])
                print(f"{name:14s} {model:7s} mean error = {err:.4f}")
    df = pd.DataFrame(rows)
    first = ["dataset", "task", "model", "rep", "fold"]
    return df[first + [c for c in df.columns if c not in first]]


def summarize(df):
    """Mean and std of the error over the 10 folds, per dataset and model."""
    return (df.groupby(["dataset", "model"])["error"]
              .agg(["mean", "std"]).round(4).reset_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(pp.DATASETS))
    parser.add_argument("--models", nargs="+", default=["linear", "ff2"], choices=list(MODEL_BUILDERS))
    parser.add_argument("--out", default=str(RESULTS_DIR / "baseline.csv"))
    parser.add_argument("--hyperparams", default=None,
                        help="best_hyperparams.json from tuning.py (default: built-in placeholders)")
    args = parser.parse_args()

    tuned = load_hyperparams(args.hyperparams) if args.hyperparams else None
    results = run_experiments(args.datasets, args.models, hyperparams=tuned)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print("\n", summarize(results).to_string(index=False))
    print(f"\nSaved {len(results)} runs to {args.out}")
