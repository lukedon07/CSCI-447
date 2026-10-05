import numpy as np
import pandas as pd
from scipy import stats
import argparse
from tree import DecisionTree
from datasetLoading import load_dataset, ALL_DATASETS



# --------------------------------------------------------------------------
# Evaluation: 20% pruning hold-out + 5x2 cv
# --------------------------------------------------------------------------
def split_two(idx, y, frac_a, rng, stratify):
    """Split idx into (a, b) with ~frac_a in a; stratified by class if requested."""
    if not stratify:
        p = rng.permutation(idx)
        k = int(round(frac_a * len(p)))
        return p[:k], p[k:]
    a, b = [], []
    for c in np.unique(y[idx]):
        g = rng.permutation(idx[y[idx] == c])
        k = int(round(frac_a * len(g)))
        a.append(g[:k]);
        b.append(g[k:])
    return np.concatenate(a), np.concatenate(b)


def metric(task, y_true, y_pred):
    if task == "classification":
        return float((y_true == y_pred).mean())  # accuracy
    return float(((y_true - y_pred) ** 2).mean())  # MSE


def five_by_two(ds, seed=0, reps=5, prune_frac=0.2):
    rng = np.random.default_rng(seed)
    strat = ds.task == "classification"
    all_idx = np.arange(len(ds))
    prune_idx, cv_idx = split_two(all_idx, ds.y, prune_frac, rng, strat)
    Xp, yp = ds.subset(prune_idx)

    r = {k: np.zeros((reps, 2)) for k in
         ("m_full", "m_pruned", "nodes_full", "nodes_pruned")}
    for i in range(reps):
        A, B = split_two(cv_idx, ds.y, 0.5, rng, strat)
        for j, (tr, te) in enumerate([(A, B), (B, A)]):
            Xtr, ytr = ds.subset(tr)
            Xte, yte = ds.subset(te)
            tree = DecisionTree(ds.task, ds.is_numeric,
                                n_classes=len(ds.class_names) if strat else None,
                                feature_names=ds.names).fit(Xtr, ytr)
            r["nodes_full"][i, j] = tree.n_nodes()
            r["m_full"][i, j] = metric(ds.task, yte, tree.predict(Xte, len(yte)))
            tree.prune(Xp, yp)
            r["nodes_pruned"][i, j] = tree.n_nodes()
            r["m_pruned"][i, j] = metric(ds.task, yte, tree.predict(Xte, len(yte)))
    return r


def paired_5x2_tests(a, b):
    """Dietterich 5x2cv t-test and Alpaydin 5x2cv F-test on a - b (5x2 arrays)."""
    d = a - b
    m = d.mean(axis=1)
    s2 = (d[:, 0] - m) ** 2 + (d[:, 1] - m) ** 2
    den = s2.mean()
    if den < 1e-18:
        return 0.0, 1.0, 0.0, 1.0
    t = d[0, 0] / np.sqrt(den)
    p_t = 2 * stats.t.sf(abs(t), 5)
    f = (d ** 2).sum() / (2 * s2.sum())
    p_f = stats.f.sf(f, 10, 5)
    return t, p_t, f, p_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="..\datasets")
    ap.add_argument("--only", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--show-tree", action="store_true",
                    help="print a pruned tree trained on all non-pruning data")
    args = ap.parse_args()

    rows = []
    for name in args.only:
        ds = load_dataset(name, args.data_dir)
        r = five_by_two(ds, seed=args.seed)
        t, p_t, f, p_f = paired_5x2_tests(r["m_full"], r["m_pruned"])
        mname = "accuracy" if ds.task == "classification" else "MSE"
        rows.append(dict(
            dataset=name, task=ds.task, metric=mname, n=len(ds),
            full_mean=r["m_full"].mean(), full_std=r["m_full"].std(ddof=1),
            pruned_mean=r["m_pruned"].mean(), pruned_std=r["m_pruned"].std(ddof=1),
            nodes_full=r["nodes_full"].mean(), nodes_pruned=r["nodes_pruned"].mean(),
            t_stat=t, p_t=p_t, F_stat=f, p_F=p_f))
        w = rows[-1]
        print(f"{name:8s} {mname:8s} full={w['full_mean']:.4f}±{w['full_std']:.4f} "
              f"pruned={w['pruned_mean']:.4f}±{w['pruned_std']:.4f} "
              f"nodes {w['nodes_full']:.0f}->{w['nodes_pruned']:.0f} "
              f"p(t)={p_t:.3f} p(F)={p_f:.3f}")
        if args.show_tree:
            rng = np.random.default_rng(args.seed)
            strat = ds.task == "classification"
            pi, ci = split_two(np.arange(len(ds)), ds.y, 0.2, rng, strat)
            Xc, yc = ds.subset(ci)
            tr = DecisionTree(ds.task, ds.is_numeric,
                              len(ds.class_names) if strat else None,
                              ds.names).fit(Xc, yc)
            tr.prune(*ds.subset(pi))
            tr.print_tree(ds.class_names)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()