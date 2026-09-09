"""

MODEL INTERFACE
---------------

    model.fit(X_train, y_train)     # X: DataFrame, y: Series
    model.predict(X_test)           # -> sequence of predictions, len == len(X_test)
    
    from evaluation import (holdout_split, five_by_two_cv, grid_search,
                            classification_error, mean_squared_error)

    # 1. carve off a tuning slice that evaluation never touches
    X_tune, y_tune, X_eval, y_eval = holdout_split(X, y, frac=0.2, stratify=True)

    # 2. pick hyperparameters using ONLY the tuning slice
    best = grid_search(lambda p: KNNClassifier(**p),
                       {"k": [1, 3, 5, 7, 9], "p": [1, 2]},
                       X_tune, y_tune, classification_error, stratify=True)

    # 3. evaluate on the untouched remainder
    result = five_by_two_cv(lambda: KNNClassifier(**best.params),
                            X_eval, y_eval, classification_error, stratify=True)
    print(result)
"""

import numpy as np
import pandas as pd
from itertools import product

try:
    from scipy import stats as _scipy_stats
except ImportError:  # scipy is optional; we just lose p-values without it
    _scipy_stats = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Every metric is "lower is better" so the tuning code can always minimize and
# never has to ask which direction it is supposed to go.

def classification_error(y_true, y_pred):
    """Fraction of predictions that are wrong. 0.0 is perfect, 1.0 is hopeless."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true != y_pred))


def accuracy(y_true, y_pred):
    """Fraction correct. Convenience for the report; do not tune on this."""
    return 1.0 - classification_error(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    """Average squared difference. The regression metric the assignment asks for."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """MSE back in the units of the target. Easier to talk about in the paper."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def epsilon_error(epsilon):
    """
    Build a regression metric that behaves like classification error.

    A regression prediction has no notion of "correct" -- predicting 9.3 when the
    truth is 9.4 is not wrong in any useful sense. Edited and condensed kNN need a
    keep-or-drop rule, so we declare a prediction correct when it lands within
    epsilon of the true value, and this returns the fraction that do not.

    Returns a function (y_true, y_pred) -> error rate, so it drops into
    five_by_two_cv and grid_search anywhere a metric is expected.

        metric = epsilon_error(0.5)
        metric(y_true, y_pred)
    """
    def _metric(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred) > epsilon))

    _metric.__name__ = f"epsilon_error(eps={epsilon})"
    return _metric


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _as_frame_series(X, y):
    """Accept numpy arrays or pandas objects, always hand back pandas."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y).ravel())
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def stratified_half_indices(y, rng):
    """
    Split row positions into two halves that preserve class proportions.

    Congressional votes is roughly 61/39. An unlucky shuffle that piles most of
    one class into a single half makes the resulting scores meaningless, so for
    classification we split each class separately and then recombine.

    Returns (first_half, second_half) as integer position arrays.
    """
    y = np.asarray(y)
    first, second = [], []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = len(idx) // 2
        # Odd class counts alternate which half gets the spare row, so neither
        # half systematically ends up larger.
        if len(idx) % 2 == 1 and rng.random() < 0.5:
            cut += 1
        first.append(idx[:cut])
        second.append(idx[cut:])

    first = np.concatenate(first) if first else np.array([], dtype=int)
    second = np.concatenate(second) if second else np.array([], dtype=int)
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


def random_half_indices(n, rng):
    """Plain shuffled 50/50 split of n row positions. Used for regression."""
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = n // 2
    return idx[:cut], idx[cut:]


def holdout_split(X, y, frac=0.2, stratify=False, seed=808):
    """
    Pull a tuning slice off the front of the data.

    Hyperparameters (k, p, gamma, epsilon) get chosen on the slice this returns
    first; 5x2 cross-validation then runs on the remainder. Tuning on data you
    later test against inflates your results, and it is the single easiest way to
    accidentally report numbers that are not real.

    Call this ONCE per dataset, at the very top of the experiment.

    Returns (X_tune, y_tune, X_eval, y_eval).
    """
    X, y = _as_frame_series(X, y)
    rng = np.random.default_rng(seed)
    n = len(X)
    n_tune = int(round(n * frac))

    if stratify:
        # Take `frac` of each class rather than `frac` of the whole set.
        tune_idx = []
        for cls in np.unique(y.values):
            idx = np.flatnonzero(y.values == cls)
            rng.shuffle(idx)
            take = max(1, int(round(len(idx) * frac)))
            tune_idx.append(idx[:take])
        tune_idx = np.concatenate(tune_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        tune_idx = idx[:n_tune]

    mask = np.zeros(n, dtype=bool)
    mask[tune_idx] = True

    return (X[mask].reset_index(drop=True), y[mask].reset_index(drop=True),
            X[~mask].reset_index(drop=True), y[~mask].reset_index(drop=True))


def five_by_two_folds(X, y, stratify=False, seed=808):
    """
    Generate the ten (train, test) index pairs of 5x2 cross-validation.

    Five times: shuffle, cut the data in half, train on A and test on B, then
    train on B and test on A. Ten scores come out.

    We use this rather than 10-fold because each half is a genuinely independent
    training set, which is what the 5x2cv paired t-test needs in order to say one
    algorithm beat another rather than got lucky on a split.

    Splits depend only on `seed`, so two different algorithms called with the same
    seed see byte-identical folds -- which is required for the paired test.

    Yields (repetition, fold, train_idx, test_idx).
    """
    X, y = _as_frame_series(X, y)

    for rep in range(5):
        # Derive each repetition's rng from the base seed so repetitions differ
        # but the whole sequence stays reproducible.
        rng = np.random.default_rng(seed + rep * 1000)

        if stratify:
            half_a, half_b = stratified_half_indices(y.values, rng)
        else:
            half_a, half_b = random_half_indices(len(X), rng)

        yield rep, 0, half_a, half_b   # train on A, test on B
        yield rep, 1, half_b, half_a   # train on B, test on A


class CVResult:
    """Ten fold scores plus the summary statistics the report needs."""

    def __init__(self, scores, metric_name="score", label=None):
        self.scores = list(scores)
        self.metric_name = metric_name
        self.label = label

    @property
    def mean(self):
        return float(np.mean(self.scores))

    @property
    def std(self):
        # ddof=1: these ten folds are a sample, not the whole population.
        return float(np.std(self.scores, ddof=1))

    def as_dict(self):
        return {
            "label": self.label,
            "metric": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "scores": self.scores,
        }

    def __repr__(self):
        name = f"{self.label}: " if self.label else ""
        return f"{name}{self.metric_name} = {self.mean:.4f} +/- {self.std:.4f} (n=10)"


def five_by_two_cv(model_factory, X, y, metric, stratify=False, seed=808,
                   label=None, verbose=False):
    """
    Run 5x2 cross-validation and return a CVResult.

    model_factory : zero-argument callable returning a FRESH untrained model.
                    Pass `lambda: KNNClassifier(k=5)`, not `KNNClassifier(k=5)` --
                    a single instance reused across folds would carry training
                    data from one fold into the next.
    metric        : callable (y_true, y_pred) -> float, lower is better.
    stratify      : True for classification, False for regression.
    seed          : same seed => same folds, so results are comparable across
                    algorithms and reproducible for the paper.
    """
    X, y = _as_frame_series(X, y)
    scores = []

    for rep, fold, train_idx, test_idx in five_by_two_folds(X, y, stratify, seed):
        model = model_factory()
        model.fit(X.iloc[train_idx].reset_index(drop=True),
                  y.iloc[train_idx].reset_index(drop=True))
        preds = model.predict(X.iloc[test_idx].reset_index(drop=True))
        score = metric(y.iloc[test_idx].values, preds)
        scores.append(score)

        if verbose:
            print(f"  rep {rep + 1} fold {fold + 1}: "
                  f"train={len(train_idx)} test={len(test_idx)} score={score:.4f}")

    metric_name = getattr(metric, "__name__", "score")
    return CVResult(scores, metric_name=metric_name, label=label)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

class TuningResult:
    def __init__(self, params, score, table):
        self.params = params        # best parameter dict
        self.score = score          # its score
        self.table = table          # DataFrame of every combination tried

    def __repr__(self):
        return f"best={self.params} score={self.score:.4f}"


def expand_grid(param_grid):
    """{'k': [1, 3], 'p': [1, 2]} -> four dicts, one per combination."""
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in product(*(param_grid[k] for k in keys))]


def grid_search(model_factory_from_params, param_grid, X, y, metric,
                stratify=False, seed=808, n_repeats=2, verbose=False):
    """
    Try every parameter combination and return the one with the lowest score.

    Run this on the TUNING SLICE ONLY -- the X_tune/y_tune that holdout_split
    returned. Never on the evaluation data.

    model_factory_from_params : callable taking a params dict, returning a model.
                                e.g. lambda p: KNNClassifier(**p)
    param_grid : dict of name -> list of values, e.g. {"k": [1,3,5], "p": [1,2]}
    n_repeats  : how many of the 5 repetitions to use while tuning. 2 is usually
                 enough and keeps tuning from dominating your runtime; bump it to
                 5 if a dataset's scores look unstable.

    The .table attribute holds every combination and its score, which is what you
    want for the "how we tuned k" plot the assignment asks for.
    """
    X, y = _as_frame_series(X, y)
    combos = expand_grid(param_grid)
    rows = []

    for params in combos:
        fold_scores = []
        for rep, fold, train_idx, test_idx in five_by_two_folds(X, y, stratify, seed):
            if rep >= n_repeats:
                break
            model = model_factory_from_params(params)
            model.fit(X.iloc[train_idx].reset_index(drop=True),
                      y.iloc[train_idx].reset_index(drop=True))
            preds = model.predict(X.iloc[test_idx].reset_index(drop=True))
            fold_scores.append(metric(y.iloc[test_idx].values, preds))

        row = dict(params)
        row["score"] = float(np.mean(fold_scores))
        row["std"] = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        rows.append(row)

        if verbose:
            print(f"  {params} -> {row['score']:.4f}")

    table = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    best = table.iloc[0]
    best_params = {k: best[k] for k in param_grid.keys()}

    # pandas hands back numpy scalars; convert so downstream code sees plain ints
    for k, v in best_params.items():
        if isinstance(v, (np.integer,)):
            best_params[k] = int(v)
        elif isinstance(v, (np.floating,)):
            best_params[k] = float(v)

    return TuningResult(best_params, float(best["score"]), table)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def five_by_two_cv_t_test(scores_a, scores_b):
    """
    Dietterich's 5x2cv paired t-test.

    Answers: is algorithm A's advantage over B real, or is it split-to-split
    noise? This is the test 5x2 CV exists to enable, and it is what lets the
    paper say "significantly better" instead of "a bit better."

    Both score lists must come from five_by_two_cv calls with the SAME seed and
    the same stratify setting, so the folds line up pairwise.

    Returns a dict with the t statistic, degrees of freedom (always 5), the
    p-value if scipy is available, and the mean difference.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != 10 or len(b) != 10:
        raise ValueError("5x2cv t-test needs exactly 10 scores from each algorithm")

    diffs = (a - b).reshape(5, 2)          # rows = repetitions, cols = the 2 folds
    means = diffs.mean(axis=1, keepdims=True)
    variances = ((diffs - means) ** 2).sum(axis=1)   # per-repetition variance

    denom = np.sqrt(variances.mean())
    if denom == 0:
        t_stat = 0.0
    else:
        # Numerator is the first difference specifically -- that is the statistic
        # as Dietterich defines it, not the overall mean.
        t_stat = float(diffs[0, 0] / denom)

    result = {
        "t_statistic": t_stat,
        "df": 5,
        "mean_difference": float(np.mean(a - b)),
        "p_value": None,
    }

    if _scipy_stats is not None:
        result["p_value"] = float(2 * (1 - _scipy_stats.t.cdf(abs(t_stat), df=5)))

    return result


def compare(result_a, result_b, alpha=0.05):
    """Human-readable wrapper around the t-test, for printing while you work."""
    test = five_by_two_cv_t_test(result_a.scores, result_b.scores)
    name_a = result_a.label or "A"
    name_b = result_b.label or "B"

    lines = [
        f"{name_a}: {result_a.mean:.4f} +/- {result_a.std:.4f}",
        f"{name_b}: {result_b.mean:.4f} +/- {result_b.std:.4f}",
        f"t({test['df']}) = {test['t_statistic']:.4f}",
    ]
    if test["p_value"] is not None:
        verdict = "significant" if test["p_value"] < alpha else "not significant"
        lines.append(f"p = {test['p_value']:.4f} ({verdict} at alpha={alpha})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def results_table(results):
    """
    Turn a list of CVResult objects into a DataFrame ready for the paper.

    Has .to_latex(), which saves retyping ten numbers into the JMLR template
    and getting one of them wrong.
    """
    rows = []
    for r in results:
        rows.append({
            "model": r.label,
            "metric": r.metric_name,
            "mean": r.mean,
            "std": r.std,
            "min": min(r.scores),
            "max": max(r.scores),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test: python evaluation.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    class _NullClassifier:
        def fit(self, X, y):
            self.prediction = Counter(y).most_common(1)[0][0]
        def predict(self, X):
            return [self.prediction] * len(X)

    class _NullRegressor:
        def fit(self, X, y):
            self.prediction = float(np.mean(y))
        def predict(self, X):
            return [self.prediction] * len(X)

    print("=" * 62)
    print("Classification: car evaluation, null model")
    print("=" * 62)
    car = pd.read_csv("datasets/car.data",
                      names=["buying", "maint", "doors", "persons",
                             "lug_boot", "safety", "class"])
    Xc, yc = car.drop("class", axis=1), car["class"]

    Xc_tune, yc_tune, Xc_eval, yc_eval = holdout_split(Xc, yc, frac=0.2, stratify=True)
    print(f"tuning slice: {len(Xc_tune)} rows | evaluation set: {len(Xc_eval)} rows")
    print(f"class balance preserved: full={yc.value_counts(normalize=True).round(3).to_dict()}")
    print(f"                         tune={yc_tune.value_counts(normalize=True).round(3).to_dict()}")

    res_c = five_by_two_cv(_NullClassifier, Xc_eval, yc_eval,
                           classification_error, stratify=True,
                           label="null classifier", verbose=True)
    print(res_c)

    print()
    print("=" * 62)
    print("Regression: abalone, null model")
    print("=" * 62)
    ab = pd.read_csv("datasets/abalone.data",
                     names=["Sex", "Length", "Diameter", "Height", "Whole weight",
                            "Shucked weight", "Visceral weight", "Shell weight", "Rings"])
    Xa, ya = ab.drop("Rings", axis=1), ab["Rings"]

    Xa_tune, ya_tune, Xa_eval, ya_eval = holdout_split(Xa, ya, frac=0.2)
    res_a = five_by_two_cv(_NullRegressor, Xa_eval, ya_eval,
                           mean_squared_error, label="null regressor")
    print(f"tuning slice: {len(Xa_tune)} rows | evaluation set: {len(Xa_eval)} rows")
    print(res_a)

    print()
    print("=" * 62)
    print("Sanity checks")
    print("=" * 62)

    # Folds must be disjoint and must cover everything.
    for rep, fold, tr, te in five_by_two_folds(Xc_eval, yc_eval, stratify=True):
        assert len(set(tr) & set(te)) == 0, "train and test overlap!"
        assert len(tr) + len(te) == len(Xc_eval), "folds do not cover the data"
    print("PASS  train/test disjoint and complete across all 10 folds")

    # The tuning slice must not leak into the evaluation set.
    assert len(Xc_tune) + len(Xc_eval) == len(Xc)
    print("PASS  tuning slice and evaluation set partition the data")

    # Same seed must give identical folds, or the paired t-test is invalid.
    r1 = five_by_two_cv(_NullClassifier, Xc_eval, yc_eval, classification_error,
                        stratify=True, seed=42)
    r2 = five_by_two_cv(_NullClassifier, Xc_eval, yc_eval, classification_error,
                        stratify=True, seed=42)
    assert r1.scores == r2.scores, "same seed produced different folds"
    print("PASS  same seed reproduces identical folds")

    # A model that is deliberately worse should be detectably worse.
    class _AlwaysWrong:
        def fit(self, X, y):
            self.classes = list(pd.Series(y).unique())
        def predict(self, X):
            return [self.classes[-1]] * len(X)

    r_bad = five_by_two_cv(_AlwaysWrong, Xc_eval, yc_eval, classification_error,
                           stratify=True, label="always minority")
    print()
    print(compare(res_c, r_bad))

    print()
    print("Grid search demo (null model ignores k, so all rows should tie):")
    tuned = grid_search(lambda p: _NullClassifier(), {"k": [1, 3, 5]},
                        Xc_tune, yc_tune, classification_error, stratify=True)
    print(tuned.table)

    print()
    print(results_table([res_c, r_bad]))
