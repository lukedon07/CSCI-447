"""Print every number needed to write the report as ONE text block.

    python report_summary.py                 # reads results/
    python report_summary.py results/quick   # or another results folder

Copy the whole output and paste it into the chat.
"""
import json
import sys
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
A = D / "analysis"
FMT = lambda v: f"{v:.4g}"


def section(title):
    print(f"\n### {title}")


def read(path):
    if not path.exists():
        print(f"(missing: {path})")
        return None
    return pd.read_csv(path)


def show(df, cols=None):
    if df is None:
        return
    cols = [c for c in (cols or df.columns) if c in df.columns]
    print(df[cols].to_string(index=False, float_format=FMT))


section("compressibility.csv")
show(read(D / "compressibility.csv"))

section("hypothesis_table.csv  (baseline vs candidate)")
show(read(A / "hypothesis_table.csv"),
     ["dataset", "C", "group", "baseline", "candidate", "baseline_error", "candidate_error", "mean_diff",
      "std_diff", "rer", "t", "p_one", "p_one_holm", "p_two", "p_wilcoxon_one", "p_F", "significant"])

section("hypothesis_verdict.json")
if (A / "hypothesis_verdict.json").exists():
    print(json.dumps(json.loads((A / "hypothesis_verdict.json").read_text(encoding="utf-8")), indent=1))
else:
    print("(missing)")

section("summary_error.csv  (mean ± std over 10 folds)")
show(read(A / "summary_error.csv"))

section("summary_secondary.csv  (accuracy / macro-F1 / R2)")
show(read(A / "summary_secondary.csv"))

section("extra_comparisons.csv  (linear vs ff2, ae_finetune vs ae_frozen)")
show(read(A / "extra_comparisons.csv"),
     ["dataset", "baseline", "candidate", "baseline_error", "candidate_error", "rer", "p_one", "p_one_holm",
      "p_two_holm"])

section("best_hyperparams.json  (tuned values used in the final runs)")
bp = D / "best_hyperparams.json"
if bp.exists():
    best = json.loads(bp.read_text(encoding="utf-8"))
    for ds, models in best.items():
        for m, hp in models.items():
            keep = {k: v for k, v in hp.items() if not k.startswith("ae_") or k == "ae_epochs"}
            print(f"{ds:14s} {m:12s} {keep}")
else:
    print("(missing)")

section("tuning_log.csv  (validation loss across the trials of each model)")
log = read(D / "tuning_log.csv")
if log is not None:
    ok = log[log["val_loss"].replace([float("inf")], float("nan")).notna()]
    g = ok.groupby(["dataset", "model"])["val_loss"].agg(trials="count", best="min", median="median", worst="max")
    g["diverged_trials"] = log.groupby(["dataset", "model"]).size() - g["trials"]
    print(g.reset_index().to_string(index=False, float_format=FMT))
    ceiling = log["best_epoch"].max()
    print(f"\n(max epochs allowed = {ceiling}; winners that used the full {ceiling} epochs, i.e. still improving:)")
    if bp.exists():
        hit = [(ds, m) for ds, ms in best.items() for m, hp in ms.items() if hp["epochs"] >= ceiling]
        print(hit if hit else "none")

section("final.csv  (run counts, reconstruction error, training time)")
fin = read(D / "final.csv")
if fin is not None:
    print(f"{len(fin)} runs; models = {sorted(fin['model'].unique())}; datasets = {sorted(fin['dataset'].unique())}")
    ae = fin[fin["model"].str.startswith("ae_")]
    if len(ae):
        print("\nreconstruction MSE (mean over folds):")
        print(ae.groupby(["dataset", "model"])[["recon_train", "recon_test"]].mean().reset_index()
              .to_string(index=False, float_format=FMT))
    print("\nmean training seconds per run:")
    print(fin.groupby(["dataset", "model"])["seconds"].mean().unstack().to_string(float_format=FMT))