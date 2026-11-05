"""Run the whole project in the correct order.

    python run_all.py --quick      # ~minutes: 3 small datasets, tiny budgets, separate folder
    python run_all.py              # the real thing: all six datasets, full tuning

Steps (each is skipped if its output file already exists, so an interrupted run
resumes where it stopped; use --force to redo everything):

  1. tests            all unit tests must pass                      (skip: --skip-tests)
  2. compressibility  C for each dataset  -> compressibility.csv    (before any model runs)
  3. tuning           best hyperparameters -> best_hyperparams.json, tuning_log.csv
  4. experiment       linear, ff2, ae_finetune, ae_frozen x 6 datasets x 10 folds -> final.csv
  5. analysis         statistics, tables, figures -> analysis/

Outputs go to results/ (or results/quick/ with --quick, or --out-dir).
Decide --baseline / --candidate BEFORE looking at any results.
"""
import argparse
import io
import json
import sys
import time
import unittest
from pathlib import Path

import analysis
import compressibility
import experiment
import preProcess as pp
import tuning

ROOT = Path(__file__).resolve().parent
MODELS = ["linear", "ff2", "ae_finetune", "ae_frozen"]
QUICK = {"datasets": ["breast_cancer", "hardware", "votes"], "trials": 4, "max_epochs": 20, "inner": 1}


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def run_tests(pattern="test_*.py", start_dir=ROOT):
    """Run the unit tests. Finding NO tests counts as a failure, since a check that
    never ran proves nothing (usually the test files are not named like the pattern)."""
    suite = unittest.defaultTestLoader.discover(str(start_dir), pattern=pattern, top_level_dir=str(start_dir))
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=0).run(suite)
    print(f"{result.testsRun} tests run, {len(result.failures)} failures, {len(result.errors)} errors")
    if result.testsRun == 0:
        print(f"No tests were found matching '{pattern}' in {start_dir}.\n"
              "Rename the test files to start with 'test_', pass --test-pattern, or use --skip-tests.")
        return False
    if not result.wasSuccessful():
        print(buffer.getvalue())
    return result.wasSuccessful()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small datasets and tiny budgets (separate output folder)")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(pp.DATASETS))
    p.add_argument("--trials", type=int, default=None, help=f"tuning trials per model (default {tuning.N_TRIALS})")
    p.add_argument("--max-epochs", type=int, default=None, help=f"max epochs while tuning (default {tuning.MAX_EPOCHS})")
    p.add_argument("--inner", type=int, default=None, help=f"inner validation splits (default {tuning.N_INNER})")
    p.add_argument("--baseline", default="ff2")
    p.add_argument("--candidate", default="ae_finetune", choices=["ae_finetune", "ae_frozen"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--test-pattern", default="test_*.py", help="file-name pattern of the unit test files")
    p.add_argument("--no-curves", action="store_true", help="skip the training-curve figures")
    a = p.parse_args(argv)

    q = QUICK if a.quick else {}
    datasets = a.datasets or q.get("datasets") or list(pp.DATASETS)
    trials = a.trials or q.get("trials") or tuning.N_TRIALS
    max_epochs = a.max_epochs or q.get("max_epochs") or tuning.MAX_EPOCHS
    inner = a.inner or q.get("inner") or tuning.N_INNER
    out = Path(a.out_dir) if a.out_dir else ROOT / "results" / ("quick" if a.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    comp_csv, best_json = out / "compressibility.csv", out / "best_hyperparams.json"
    final_csv = out / "final.csv"
    t0 = time.time()

    print(f"datasets: {datasets}\ntuning: {trials} trials, up to {max_epochs} epochs, {inner} inner splits"
          f"\nhypothesis: {a.baseline} (baseline) vs {a.candidate} (candidate)\noutput folder: {out}")

    # # 1. tests ----------------------------------------------------------
    # if not a.skip_tests:
    #     banner("1/5  Unit tests")
    #     if not run_tests(a.test_pattern):
    #         print("Tests failed - fix them before running experiments (or use --skip-tests).")
    #         return 1

    # 2. compressibility --------------------------------------------------
    banner("2/5  Compressibility (recorded before any model is trained)")
    if comp_csv.exists() and not a.force:
        print(f"{comp_csv.name} exists - reusing it (use --force to recompute)")
    else:
        compressibility.compressibility_table().to_csv(comp_csv, index=False)
    import pandas as pd
    print(pd.read_csv(comp_csv).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3. tuning -------------------------------------------------------------
    banner("3/5  Hyperparameter tuning")
    if best_json.exists() and not a.force:
        print(f"{best_json.name} exists - skipping (use --force to retune)")
    else:
        tuning.run_tuning(datasets, MODELS, out_dir=out, n_trials=trials, n_inner=inner, max_epochs=max_epochs)
    tuned = experiment.load_hyperparams(best_json)

    # 4. experiments ------------------------------------------------------------
    banner("4/5  5x2 cross-validation runs")
    if final_csv.exists() and not a.force:
        print(f"{final_csv.name} exists - skipping (use --force to rerun)")
        results = pd.read_csv(final_csv)
    else:
        results = experiment.run_experiments(datasets, MODELS, hyperparams=tuned)
        results.to_csv(final_csv, index=False)
        print(f"\nSaved {len(results)} runs to {final_csv}")
    print(experiment.summarize(results).to_string(index=False))

    # 5. analysis -----------------------------------------------------------------
    banner("5/5  Statistics, tables and figures")
    if results["dataset"].nunique() < 3:
        print("The hypothesis analysis needs at least 3 datasets - skipping this step.")
    else:
        table, verdict = analysis.run_analysis(final_csv, comp_csv, out / "analysis", a.baseline,
                                               a.candidate, tuned, curves=not a.no_curves)
        show = ["dataset", "C", "group", "baseline_error", "candidate_error", "rer", "p_one_holm"]
        print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nVerdict:", json.dumps({k: verdict[k] for k in
              ("H1a", "H1b", "H1c", "spearman_rho", "verdict")}, indent=2, default=str))

    banner(f"Done in {(time.time() - t0) / 60:.1f} minutes. Results are in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())