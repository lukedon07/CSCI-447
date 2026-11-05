"""Statistical comparison of two models over the 5x2 cross-validation results.

Definitions (match the hypothesis)
----------------------------------
* error            : 1 - accuracy (classification) or RMSE (regression); lower is better.
* D[i, j]          : baseline error - candidate error on repetition i, fold j.
                     Positive means the candidate (autoencoder network) is better.
* RER              : mean(D) / mean(baseline error), the relative error reduction.
* Primary test     : Dietterich's 5x2 cv paired t-test (5 degrees of freedom).
* Sensitivity tests: Alpaydin's 5x2 cv F-test and the Wilcoxon signed-rank test.
* One-sided p-value for the directional claim "candidate is better" (D > 0);
  two-sided p-values are reported too, since the candidate could be worse.
* Holm's step-down adjustment is applied across datasets.

Requires scipy.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

N_REPS = 5


# --------------------------------------------------------------------------
# Paired differences
# --------------------------------------------------------------------------
def paired_differences(df, dataset, baseline, candidate, metric="error"):
    """5x2 array D[rep, fold] = baseline - candidate, aligned fold by fold."""
    sub = df[df["dataset"] == dataset]
    a = sub[sub["model"] == baseline].set_index(["rep", "fold"])[metric]
    b = sub[sub["model"] == candidate].set_index(["rep", "fold"])[metric]
    idx = pd.MultiIndex.from_product([range(N_REPS), range(2)])
    if len(a) != 10 or len(b) != 10 or set(a.index) != set(idx) or set(b.index) != set(idx):
        raise ValueError(f"Need exactly the 10 (rep, fold) results for both '{baseline}' and "
                         f"'{candidate}' on '{dataset}'")
    return (a.loc[idx] - b.loc[idx]).to_numpy(dtype=float).reshape(N_REPS, 2)


# --------------------------------------------------------------------------
# Tests on a 5x2 difference matrix
# --------------------------------------------------------------------------
def _variances(D):
    p_bar = D.mean(axis=1)
    return ((D - p_bar[:, None]) ** 2).sum(axis=1)  # s_i^2 for each repetition


def five_by_two_t(D):
    """Dietterich (1998): t = D[0,0] / sqrt(mean(s_i^2)), 5 df.
    Returns (t, p_two_sided, p_one_sided for D > 0)."""
    D = np.asarray(D, dtype=float)
    num = D[0, 0]
    denom = np.sqrt(_variances(D).mean())
    if denom == 0:
        if num == 0:
            return 0.0, 1.0, 0.5
        return float(np.sign(num) * np.inf), 0.0, 0.0 if num > 0 else 1.0
    t = num / denom
    return float(t), float(2 * stats.t.sf(abs(t), 5)), float(stats.t.sf(t, 5))


def five_by_two_f(D):
    """Alpaydin (1999): f = sum(D^2) / (2 sum(s_i^2)), (10, 5) df. Returns (f, p)."""
    D = np.asarray(D, dtype=float)
    s = _variances(D).sum()
    if s == 0:
        return (np.inf, 0.0) if np.any(D != 0) else (np.nan, 1.0)
    f = (D ** 2).sum() / (2 * s)
    return float(f), float(stats.f.sf(f, 10, 5))


def wilcoxon_p(D):
    """Wilcoxon signed-rank on the 10 differences. Returns (p_two_sided, p_one_sided D > 0)."""
    d = np.asarray(D, dtype=float).ravel()
    if np.all(d == 0):
        return 1.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = stats.wilcoxon(d, alternative="two-sided").pvalue
        one = stats.wilcoxon(d, alternative="greater").pvalue
    return float(two), float(one)


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, returned in the original order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------
# Comparing two models over datasets
# --------------------------------------------------------------------------
def compare_models(df, baseline, candidate, datasets=None, metric="error"):
    """One row per dataset: errors, mean difference, RER and all test results."""
    datasets = sorted(df["dataset"].unique()) if datasets is None else list(datasets)
    rows = []
    for name in datasets:
        D = paired_differences(df, name, baseline, candidate, metric)
        base_mean = float(df[(df.dataset == name) & (df.model == baseline)][metric].mean())
        cand_mean = float(df[(df.dataset == name) & (df.model == candidate)][metric].mean())
        t, p_two, p_one = five_by_two_t(D)
        f, p_f = five_by_two_f(D)
        w_two, w_one = wilcoxon_p(D)
        rows.append({
            "dataset": name, "baseline": baseline, "candidate": candidate,
            "baseline_error": base_mean, "candidate_error": cand_mean,
            "mean_diff": float(D.mean()), "std_diff": float(D.std(ddof=1)),
            "rer": float(D.mean() / base_mean) if base_mean != 0 else np.nan,
            "t": t, "p_two": p_two, "p_one": p_one,
            "F": f, "p_F": p_f, "p_wilcoxon_two": w_two, "p_wilcoxon_one": w_one,
        })
    out = pd.DataFrame(rows)
    out["p_one_holm"] = holm_adjust(out["p_one"])
    out["p_two_holm"] = holm_adjust(out["p_two"])
    return out


# --------------------------------------------------------------------------
# Rank correlation between compressibility C and RER (H1c)
# --------------------------------------------------------------------------
def spearman_exact(x, y):
    """Spearman rho and its EXACT one-sided permutation p-value (H1: rho > 0).
    Enumerates every ordering of y, so it is valid for the tiny n (6) used here."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("Need two arrays of equal length >= 3")
    if n > 9:
        raise ValueError("Exact enumeration is limited to n <= 9")
    rx, ry = stats.rankdata(x), stats.rankdata(y)

    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return np.nan if denom == 0 else float((a * b).sum() / denom)

    observed = rho(rx, ry)
    if np.isnan(observed):
        return np.nan, np.nan
    perms = [rho(rx, np.array(p)) for p in itertools.permutations(ry)]
    p_one = float(np.mean(np.array(perms) >= observed - 1e-12))
    return observed, p_one


# --------------------------------------------------------------------------
# Hypothesis decision
# --------------------------------------------------------------------------
def evaluate_hypothesis(table, alpha=0.05, rho_threshold=0.83):
    """Apply the pre-registered decision rules to a per-dataset table with columns
    group, C, rer, mean_diff, p_one_holm.

    significant improvement : mean_diff > 0 and Holm-adjusted one-sided p < alpha
    H1a supported : every compressible dataset shows a significant improvement
    H1b supported : no incompressible dataset shows a significant improvement
    H1c supported : Spearman rho(C, RER) >= rho_threshold
    refuted       : the autoencoder wins significantly on an incompressible dataset but on
                    NO compressible one, or rho <= 0
    supported     : H1a, H1b and H1c all hold;  otherwise partially supported
    """
    t = table.copy()
    t["significant"] = (t["mean_diff"] > 0) & (t["p_one_holm"] < alpha)
    comp = t[t["group"] == "compressible"]
    incomp = t[t["group"] == "incompressible"]
    rho, p_rho = spearman_exact(t["C"], t["rer"])

    h1a = bool(len(comp) > 0 and comp["significant"].all())
    h1b = bool(not incomp["significant"].any())
    h1c = bool(rho >= rho_threshold)
    refuted = bool((incomp["significant"].any() and not comp["significant"].any()) or rho <= 0)
    verdict = "refuted" if refuted else ("supported" if (h1a and h1b and h1c) else "partially supported")
    return {"H1a": h1a, "H1b": h1b, "H1c": h1c,
            "n_compressible": len(comp), "n_compressible_significant": int(comp["significant"].sum()),
            "n_incompressible": len(incomp), "n_incompressible_significant": int(incomp["significant"].sum()),
            "spearman_rho": rho, "spearman_p_one_sided": p_rho,
            "alpha": alpha, "rho_threshold": rho_threshold, "verdict": verdict}
