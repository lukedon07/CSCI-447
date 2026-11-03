"""5x2 cross-validation splits.

Five repetitions; in each one the data is shuffled and cut into two halves A
and B. Fold 1 trains on A and tests on B, fold 2 trains on B and tests on A,
giving 10 (train, test) pairs. Classification splits are stratified, so every
class is divided as evenly as possible between the two halves.

The splits depend only on the labels, the task and the seed, so the same call
always returns the same folds. Every model must use the same call (same seed)
so the paired statistical tests compare models on identical folds.

    splits = five_by_two_splits(y, task)
    for i, (train_idx, test_idx) in enumerate(splits):
        rep, fold = divmod(i, 2)
"""
import numpy as np

DEFAULT_SEED = 808


def five_by_two_splits(y, task, n_reps=5, seed=DEFAULT_SEED):
    y = np.asarray(y)
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 samples to split")
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")

    rng = np.random.default_rng(seed)
    if task == "classification":
        groups = [np.where(y == c)[0] for c in np.unique(y)]
    else:
        groups = [np.arange(n)]

    splits = []
    for _ in range(n_reps):
        half = np.zeros(n, dtype=int)
        for idx in groups:
            idx = rng.permutation(idx)
            # alternate A/B down the shuffled class; a random start spreads the
            # odd leftover samples across the two halves
            half[idx] = (np.arange(len(idx)) + rng.integers(2)) % 2
        a, b = np.where(half == 0)[0], np.where(half == 1)[0]
        splits.append((a, b))
        splits.append((b, a))
    return splits