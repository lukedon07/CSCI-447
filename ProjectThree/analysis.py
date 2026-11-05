"""Turn experiment results into the tables, statistics and figures for the report.

Inputs
  results CSV          written by experiment.py (all models you want compared)
  compressibility CSV  written by compressibility.py

Outputs (in --out-dir)
  summary_error.csv / .tex     mean +- std error per dataset and model
  summary_secondary.csv        accuracy / macro-F1 (classification) and R^2 (regression)
  hypothesis_table.csv / .tex  per-dataset C, errors, RER, p-values (raw and Holm)
  hypothesis_verdict.json      H1a / H1b / H1c and the overall verdict
  extra_comparisons.csv        linear vs ff2, and ae_finetune vs ae_frozen (if present)
  fig_rer_vs_C.png, fig_rer_by_dataset.png, fig_reconstruction.png, fig_curves_<dataset>.png

    python analysis.py --results results/final.csv --baseline ff2 --candidate ae_finetune
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crossValidation as cv
import experiment as ex
import preProcess as pp
import stats as st

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GROUP_COLORS = {"compressible": "tab:blue", "incompressible": "tab:orange"}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def summary_long(df, metric="error"):
    """Mean and std (over the 10 folds) of one metric per dataset and model."""
    return (df.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index())


def pivot_mean_std(df, metric="error", models=None):
    """Datasets as rows, models as columns, cells like '0.0286 ± 0.0094'."""
    long = summary_long(df, metric)
    models = models or list(dict.fromkeys(df["model"]))
    long["cell"] = [f"{m:.4g} ± {s:.2g}" for m, s in zip(long["mean"], long["std"])]
    table = long.pivot(index="dataset", columns="model", values="cell")
    return table.loc[[d for d in pp.DATASETS if d in table.index], [m for m in models if m in table.columns]]


def secondary_table(df):
    """Accuracy and macro-F1 for classification datasets, R^2 for regression ones."""
    cls = df[df.task == "classification"].groupby(["dataset", "model"])[["accuracy", "macro_f1"]].mean()
    reg = df[df.task == "regression"].groupby(["dataset", "model"])[["r2"]].mean()
    return pd.concat([cls, reg]).reset_index().round(4)


def hypothesis_table(df, comp, baseline="ff2", candidate="ae_finetune"):
    """Per-dataset comparison joined with compressibility C and its group."""
    cmp_ = st.compare_models(df, baseline, candidate)
    table = cmp_.merge(comp[["dataset", "C", "group"]], on="dataset", how="left")
    if table["C"].isna().any():
        raise ValueError("Compressibility is missing for: "
                         f"{table[table.C.isna()].dataset.tolist()}")
    alpha = 0.05
    table["significant"] = (table["mean_diff"] > 0) & (table["p_one_holm"] < alpha)
    return table.sort_values("C", ascending=False).reset_index(drop=True)


def write_latex(table, path, caption, label, float_format="%.4g"):
    """Write a LaTeX table that is pure ASCII, so it saves and compiles on any system."""
    text = table.to_latex(index=False, caption=caption, label=label,
                          float_format=float_format, escape=True)
    text = text.replace("\u00b1", "$\\pm$").replace("\u2212", "-")
    Path(path).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Training curves (rerun one fold to record per-epoch loss)
# --------------------------------------------------------------------------
def training_curves(dataset, models=("ff2", "ae_finetune"), hyperparams=None,
                    seed=cv.DEFAULT_SEED):
    """Train each model on repetition 0, fold 0. Returns {model: {train, test, pretrain}}.
    The test curve is for plotting only; nothing is selected with it."""
    hyperparams = hyperparams or {}
    X, y, task, k = pp.load_dataset(dataset)
    tr, te = cv.five_by_two_splits(y, task, seed=seed)[0]
    Xtr, Xte, ytr, yte, _ = pp.preprocess_fold(X[tr], X[te], y[tr], y[te], task, k)
    n_out = k if task == "classification" else 1
    out = {}
    for model in models:
        hp = ex._hp_for(hyperparams, dataset, model)
        net = ex.MODEL_BUILDERS[model](Xtr.shape[1], n_out, task, hp, seed, Xtr=Xtr)
        hist = net.fit(Xtr, ytr, epochs=hp["epochs"], lr=hp["lr"], batch_size=hp["batch_size"],
                       momentum=hp["momentum"], X_val=Xte, Y_val=yte)
        pre = getattr(net, "pretrain_history", None)
        out[model] = {"train": hist["train_loss"], "test": hist["val_loss"],
                      "pretrain": pre["train_loss"] if pre else None}
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _pct(rer):
    """RER in percent for plotting; an infinite RER (perfect baseline) is drawn at -100."""
    return np.where(np.isfinite(rer), 100 * np.asarray(rer, dtype=float), -100.0)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_rer_vs_compressibility(table, verdict, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    pct = _pct(table["rer"])
    for (_, r), y in zip(table.iterrows(), pct):
        ax.scatter(r["C"], y, s=60, color=GROUP_COLORS[r["group"]], zorder=3)
        ax.annotate(r["dataset"], (r["C"], y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0.80, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Compressibility C")
    ax.set_ylabel("Relative error reduction (%)")
    ax.set_title(f"Spearman ρ = {verdict['spearman_rho']:.2f}  (n = {len(table)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rer_by_dataset(table, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [GROUP_COLORS[g] for g in table["group"]]
    bars = ax.bar(table["dataset"], _pct(table["rer"]), color=colors)
    for bar, sig in zip(bars, table["significant"]):
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*", ha="center", va="bottom")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Relative error reduction (%)")
    ax.set_title("Autoencoder network vs plain network (datasets ordered by C; * = Holm p < 0.05)")
    ax.title.set_fontsize(9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_reconstruction(df, path, model="ae_finetune"):
    plt = _plt()
    sub = df[df.model == model].groupby("dataset")[["recon_train", "recon_test"]].mean()
    sub = sub.loc[[d for d in pp.DATASETS if d in sub.index]]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, sub["recon_train"], 0.4, label="train fold")
    ax.bar(x + 0.2, sub["recon_test"], 0.4, label="test fold")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=30, ha="right")
    ax.set_ylabel("Reconstruction MSE (standardized features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_curves(curves, dataset, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for model, c in curves.items():
        axes[0].plot(c["train"], label=f"{model} train")
        axes[0].plot(c["test"], ls="--", label=f"{model} test")
        if c["pretrain"] is not None:
            axes[1].plot(c["pretrain"], label=f"{model} autoencoder")
    axes[0].set_title(f"{dataset}: prediction loss (rep 0, fold 0)")
    axes[1].set_title("Autoencoder reconstruction loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_analysis(results_csv, comp_csv, out_dir=RESULTS_DIR / "analysis", baseline="ff2",
                 candidate="ae_finetune", hyperparams=None, plots=True, curves=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    comp = pd.read_csv(comp_csv)

    pivot = pivot_mean_std(df)
    pivot.to_csv(out / "summary_error.csv")
    write_latex(pivot.reset_index(), out / "summary_error.tex",
                "Mean test error ($\\pm$ std) over the ten 5x2 cv folds (1 - accuracy, or RMSE).",
                "tab:error")
    secondary_table(df).to_csv(out / "summary_secondary.csv", index=False)

    table = hypothesis_table(df, comp, baseline, candidate)
    verdict = st.evaluate_hypothesis(table)
    table.to_csv(out / "hypothesis_table.csv", index=False)
    cols = ["dataset", "C", "group", "baseline_error", "candidate_error", "rer",
            "t", "p_one", "p_one_holm", "p_wilcoxon_one"]
    write_latex(table[cols], out / "hypothesis_table.tex",
                f"{candidate} vs {baseline}: relative error reduction (RER) and one-sided "
                "5x2 cv t-test p-values (raw and Holm-adjusted).", "tab:hypothesis")
    with open(out / "hypothesis_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    extras = []
    models = set(df["model"])
    for base, cand in (("linear", baseline), (candidate, "ae_frozen")):
        if base in models and cand in models and base != cand:
            extras.append(st.compare_models(df, base, cand))
    if extras:
        pd.concat(extras).to_csv(out / "extra_comparisons.csv", index=False)

    if plots:
        plot_rer_vs_compressibility(table, verdict, out / "fig_rer_vs_C.png")
        plot_rer_by_dataset(table, out / "fig_rer_by_dataset.png")
        if df[df.model == candidate]["recon_test"].notna().any():
            plot_reconstruction(df, out / "fig_reconstruction.png", candidate)
        if curves:
            for name in pp.DATASETS:
                if name in set(df["dataset"]):
                    c = training_curves(name, (baseline, candidate), hyperparams)
                    plot_training_curves(c, name, out / f"fig_curves_{name}.png")
    return table, verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--compressibility", default=str(RESULTS_DIR / "compressibility.csv"))
    p.add_argument("--out-dir", default=str(RESULTS_DIR / "analysis"))
    p.add_argument("--baseline", default="ff2")
    p.add_argument("--candidate", default="ae_finetune")
    p.add_argument("--hyperparams", default=None, help="best_hyperparams.json for the curve plots")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-curves", action="store_true", help="skip retraining for the curve plots")
    a = p.parse_args()
    tuned = ex.load_hyperparams(a.hyperparams) if a.hyperparams else None
    table, verdict = run_analysis(a.results, a.compressibility, a.out_dir, a.baseline, a.candidate,
                                  tuned, plots=not a.no_plots, curves=not a.no_curves)
    show = ["dataset", "C", "group", "baseline_error", "candidate_error", "rer", "p_one", "p_one_holm"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n", json.dumps(verdict, indent=2, default=str))