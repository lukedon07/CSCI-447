"""Compressibility C of each dataset (used by the hypothesis; needs no labels).

C = share of total variance captured by the first e = ceil(d/2) principal
components of the standardized training features, averaged over the 10
training folds of the 5x2 cross-validation.

    uncorrelated features -> C close to 0.5
    redundant features    -> C close to 1

A dataset is called "compressible" if C >= THRESHOLD. The threshold is fixed
here, before any model is trained.

    python compressibility.py        # prints a table and writes results/compressibility.csv

  example row:
    breast_cancer  9  5 0.898  0.004   compressible
    out of the 9 features, the first five principal components capture about 90% of its variance
    .004 is the standard deviation of C across the 10 training folds, .004 means that the numbers change very little between folds which is good
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import crossValidation as cv
import preProcess as pp

THRESHOLD = 0.80
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def variance_share(X, e):
    """Fraction of total variance in the top-e principal components of X."""
    eigvals = np.linalg.eigvalsh(np.cov(np.asarray(X, dtype=float), rowvar=False))
    eigvals = np.clip(np.sort(eigvals)[::-1], 0.0, None)
    total = eigvals.sum()
    return 1.0 if total == 0 else float(eigvals[:e].sum() / total)


def dataset_compressibility(name, seed=cv.DEFAULT_SEED):
    X, y, task, k = pp.load_dataset(name)
    d = X.shape[1]
    e = math.ceil(d / 2)
    shares = []
    for tr, te in cv.five_by_two_splits(y, task, seed=seed):
        Xtr, _, _, _, _ = pp.preprocess_fold(X[tr], X[te], y[tr], y[te], task, k)
        shares.append(variance_share(Xtr, e))
    c = float(np.mean(shares))
    return {"dataset": name, "n_features": d, "e": e, "C": c,
            "C_std": float(np.std(shares)),
            "group": "compressible" if c >= THRESHOLD else "incompressible"}


def compressibility_table():
    return pd.DataFrame([dataset_compressibility(n) for n in pp.DATASETS])


if __name__ == "__main__":
    table = compressibility_table()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "compressibility.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'compressibility.csv'}")
