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
# all of these are lower-is-better so the tuning code can just minimize

def classification_error(y_true, y_pred):
    """fraction of predictions that are wrong. 0 = perfect."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true != y_pred))


def accuracy(y_true, y_pred):
    """fraction correct. just for the writeup, don't tune on this one."""
    return 1.0 - classification_error(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    """average squared error. the regression metric the assignment asks for."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """same thing but back in the target's units, easier to talk about."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def epsilon_error(epsilon):
    """
    makes a regression metric that acts like classification error.

    there's no such thing as a "correct" regression prediction, 9.3 vs 9.4
    isn't really wrong. but edited/condensed need some keep-or-drop rule, so
    count it correct if it's within epsilon. returns the fraction that aren't.

    gives back a function, so it plugs into five_by_two_cv and grid_search
    wherever a metric goes:

        metric = epsilon_error(0.5)
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
    """takes numpy or pandas, always hands back pandas."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y).ravel())
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def stratified_half_indices(y, rng):
    """
    splits into two halves while keeping the class proportions the same.

    votes is about 61/39, so a bad shuffle could dump most of one class into
    one half and the scores would be garbage. splitting each class separately
    and recombining avoids that.

    returns (first_half, second_half) as index arrays.
    """
    y = np.asarray(y)
    first, second = [], []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = len(idx) // 2
        # odd counts, flip a coin for who gets the extra row so one half
        # doesn't always end up bigger
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
    """plain shuffled 50/50 split. used for the regression sets."""
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = n // 2
    return idx[:cut], idx[cut:]


def holdout_split(X, y, frac=0.2, stratify=False, seed=808):
    """
    pulls off a chunk of data to tune on.

    k, p, gamma and epsilon all get picked on the tuning slice, then the 5x2
    runs on whatever's left. if we tune on data we later test against the
    results come out looking better than they are.

    call this once per dataset, at the top.

    returns (X_tune, y_tune, X_eval, y_eval).
    """
    X, y = _as_frame_series(X, y)
    rng = np.random.default_rng(seed)
    n = len(X)
    n_tune = int(round(n * frac))

    if stratify:
        # take frac of each class instead of frac of the whole thing
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
    makes the ten (train, test) index pairs for 5x2 cross validation.

    five times: shuffle, cut in half, train on A test on B, then train on B
    test on A. ten scores total.

    using this instead of 10-fold because each half is an independent training
    set, which is what the paired t-test needs to tell a real difference from
    a lucky split.

    splits only depend on the seed, so two algorithms with the same seed get
    the exact same folds. the paired test breaks otherwise.

    yields (repetition, fold, train_idx, test_idx).
    """
    X, y = _as_frame_series(X, y)

    for rep in range(5):
        # each rep gets its own rng off the base seed, so the reps differ but
        # the whole thing still reproduces
        rng = np.random.default_rng(seed + rep * 1000)

        if stratify:
            half_a, half_b = stratified_half_indices(y.values, rng)
        else:
            half_a, half_b = random_half_indices(len(X), rng)

        yield rep, 0, half_a, half_b   # train on A, test on B
        yield rep, 1, half_b, half_a   # train on B, test on A


class CVResult:
    """the ten fold scores plus the summary stats for the writeup."""

    def __init__(self, scores, metric_name="score", label=None, stats=None):
        self.scores = list(scores)
        self.metric_name = metric_name
        self.label = label
        # anything extra pulled off each fitted model, one value per fold.
        # used for the edited/condensed reduction ratios.
        self.stats = list(stats) if stats else []

    @property
    def mean(self):
        return float(np.mean(self.scores))

    @property
    def std(self):
        # ddof=1, ten folds is a sample not the whole population
        return float(np.std(self.scores, ddof=1))

    @property
    def stat_mean(self):
        """average of whatever per_fold_stat collected, None if nothing was."""
        clean = [s for s in self.stats if s is not None]
        return float(np.mean(clean)) if clean else None

    @property
    def stat_std(self):
        clean = [s for s in self.stats if s is not None]
        return float(np.std(clean, ddof=1)) if len(clean) > 1 else None

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
                   label=None, verbose=False, per_fold_stat=None):
    """
    runs the 5x2 and gives back a CVResult.

    model_factory : function that returns a NEW untrained model, so
                    lambda: KNNClassifier(k=5) and not KNNClassifier(k=5).
                    reusing one instance would drag fold 1's training data
                    into fold 2.
    metric        : (y_true, y_pred) -> float, lower is better.
    stratify      : True for classification, False for regression.
    seed          : same seed means same folds, so runs are comparable and
                    reproduce later.
    per_fold_stat : optional function(fitted_model) -> number, called after
                    each fold. the models get thrown away otherwise, so this
                    is the only chance to grab anything off them. used for
                    the edited/condensed reduction ratio.
    """
    X, y = _as_frame_series(X, y)
    scores = []
    stats = []

    for rep, fold, train_idx, test_idx in five_by_two_folds(X, y, stratify, seed):
        model = model_factory()
        model.fit(X.iloc[train_idx].reset_index(drop=True),
                  y.iloc[train_idx].reset_index(drop=True))
        preds = model.predict(X.iloc[test_idx].reset_index(drop=True))
        score = metric(y.iloc[test_idx].values, preds)
        scores.append(score)

        if per_fold_stat is not None:
            try:
                stats.append(per_fold_stat(model))
            except Exception:
                # a model that doesn't support the stat just gets nothing,
                # no reason to kill the whole run over it
                stats.append(None)

        if verbose:
            print(f"  rep {rep + 1} fold {fold + 1}: "
                  f"train={len(train_idx)} test={len(test_idx)} score={score:.4f}")

    metric_name = getattr(metric, "__name__", "score")
    return CVResult(scores, metric_name=metric_name, label=label, stats=stats)


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
    """{'k': [1, 3], 'p': [1, 2]} turns into four dicts, one per combo."""
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in product(*(param_grid[k] for k in keys))]


def grid_search(model_factory_from_params, param_grid, X, y, metric,
                stratify=False, seed=808, n_repeats=2, verbose=False):
    """
    tries every parameter combo and returns whichever scored lowest.

    only run this on the tuning slice from holdout_split, never on the
    evaluation data.

    model_factory_from_params : function taking a params dict and returning a
                                model, like lambda p: KNNClassifier(**p)
    param_grid : name -> list of values, like {"k": [1,3,5], "p": [1,2]}
    n_repeats  : how many of the 5 reps to use while tuning. 2 is usually fine
                 and keeps tuning from taking forever. bump to 5 if a dataset
                 looks unstable.

    .table has every combo and its score, which is what the k-tuning plots
    come from.
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

    # pandas gives back numpy scalars, convert so the models get plain ints
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

    tells us whether A actually beat B or just got lucky on the splits. this
    is the whole reason for doing 5x2 in the first place, and it's what lets
    the paper say "significantly better" instead of just "better".

    both score lists have to come from five_by_two_cv with the same seed and
    same stratify setting, otherwise the folds don't line up pairwise.

    returns the t statistic, df (always 5), the p-value if scipy is around,
    and the mean difference.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != 10 or len(b) != 10:
        raise ValueError("5x2cv t-test needs exactly 10 scores from each algorithm")

    diffs = (a - b).reshape(5, 2)

    # For each repetition, calculate the mean difference between
    # the two train/test folds.
    rep_means = diffs.mean(axis=1)

    # Dietterich's 5x2cv variance estimate:
    #
    # s_i^2 = (d_i1 - mean_i)^2 + (d_i2 - mean_i)^2
    #
    # There are only two observations per repetition, so this is
    # equivalent to the sample variance because n - 1 = 1.
    rep_variances = (
            (diffs[:, 0] - rep_means) ** 2
            + (diffs[:, 1] - rep_means) ** 2
    )

    # Average the five repetition-specific variance estimates.
    variance_estimate = np.mean(rep_variances)

    denom = np.sqrt(variance_estimate)

    if denom == 0:
        t_stat = 0.0
    else:
        # Dietterich uses the first fold of the first repetition
        # as the numerator.
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
    """readable version of the t-test output, handy for printing while working."""
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
    turns a list of CVResults into a DataFrame for the paper.

    it has .to_latex() so the numbers don't have to get retyped into the
    template by hand.
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
# quick self-test, just run: python evaluation.py
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
