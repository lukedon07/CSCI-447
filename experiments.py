# KNN must return one prediction per input row.
import os
import pandas as pd
from pathlib import Path
from data import load_dataset
from models import (
    NullClassifier,
    NullRegressor,
    KNNClassifier,
    KNNRegressor,
    EditedKNNClassifier,
    EditedKNNRegressor,
    CondensedKNNClassifier,
    CondensedKNNRegressor,
)
from evaluation import(
    holdout_split,
    grid_search,
    five_by_two_cv,
    classification_error,
    mean_squared_error,
    epsilon_error
)

def make_classifier(config, **params):
    return KNNClassifier(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )


def make_regressor(config, **params):
    return KNNRegressor(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )

def make_edited_classifier(config, **params):
    return EditedKNNClassifier(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )


def make_condensed_classifier(config, **params):
    return CondensedKNNClassifier(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )


def make_edited_regressor(config, **params):
    return EditedKNNRegressor(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )


def make_condensed_regressor(config, **params):
    return CondensedKNNRegressor(
        **params,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        cyclic_features=config.cyclic_features,
    )


# ============================================================
# Configuration
# ============================================================

DATASET_NAMES = [
    "breast_cancer",
    "car",
    "votes",
    "abalone",
    "computer_hardware",
    "forest_fires",
]


CLASSIFICATION_DATASETS = {
    "breast_cancer",
    "car",
    "votes",
}


REGRESSION_DATASETS = {
    "abalone",
    "computer_hardware",
    "forest_fires",
}


RESULTS_DIR = Path("results")


# ============================================================
# Hyperparameter grids
# ============================================================

CLASSIFIER_GRID = {
    "k": [1, 3, 5, 7, 9, 15, 21],
    "p": [1, 2],
}


REGRESSOR_GRID = {
    "k": [1, 3, 5, 7, 9, 15, 21],
    "p": [1, 2],
    "gamma": [0.1, 0.5, 1.0, 2.0, 5.0],
}


# Epsilon is expressed in transformed target units for Forest Fires.
REGRESSION_EPSILONS = {
    "abalone": [0.25, 0.5, 1.0, 1.5, 2.0],
    "computer_hardware": [5.0, 10.0, 20.0, 40.0, 80.0],
    "forest_fires": [0.05, 0.10, 0.25, 0.50, 1.00],
}


# ============================================================
# Utility functions
# ============================================================

def make_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_classification(dataset_name):
    return dataset_name in CLASSIFICATION_DATASETS


def is_regression(dataset_name):
    return dataset_name in REGRESSION_DATASETS


# ============================================================
# Dataset tuning
# ============================================================

def tune_classifier(config, X_tune, y_tune):
    """
    Tune k and p using ONLY the tuning partition.
    """

    result = grid_search(
        lambda params: KNNClassifier(
            **params,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
        ),
        CLASSIFIER_GRID,
        X_tune,
        y_tune,
        classification_error,
        stratify=True,
        seed=808,
        n_repeats=2,
    )

    return result


def tune_regressor(config, X_tune, y_tune):
    """
    Tune k, p, and gamma using ONLY the tuning partition.
    """

    result = grid_search(
        lambda params: KNNRegressor(
            **params,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
        ),
        REGRESSOR_GRID,
        X_tune,
        y_tune,
        mean_squared_error,
        stratify=False,
        seed=808,
        n_repeats=2,
    )

    return result


def tune_epsilon(dataset_name, config, X_tune, y_tune):
    """
    Tune epsilon independently for each regression dataset.

    epsilon is needed only for determining whether a training example
    should be retained during editing/condensing.
    """

    best_epsilon = None
    best_score = float("inf")
    rows = []

    for epsilon in REGRESSION_EPSILONS[dataset_name]:

        metric = epsilon_error(epsilon)

        result = five_by_two_cv(
            lambda: KNNRegressor(
                k=1,
                p=2,
                gamma=1.0,
                numeric_features=config.numeric_features,
                categorical_features=config.categorical_features,
                cyclic_features=config.cyclic_features,
            ),
            X_tune,
            y_tune,
            metric,
            stratify=False,
            seed=808,
        )

        score = result.mean

        rows.append({
            "epsilon": epsilon,
            "score": score,
        })

        if score < best_score:
            best_score = score
            best_epsilon = epsilon

    table = pd.DataFrame(rows)

    return best_epsilon, table


# ============================================================
# Standard KNN experiments
# ============================================================

def run_classification_models(dataset_name, config, X_eval, y_eval, params):

    """
    Evaluate classification models using the already-tuned parameters.
    """

    results = []

    k = int(params["k"])
    p = float(params["p"])

    models = {
        "Null": lambda: NullClassifier(),

        "KNN": lambda: KNNClassifier(
            k=k,
            p=p,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
            random_state=808,
        ),

        "Edited KNN": lambda: EditedKNNClassifier(
            k=k,
            p=p,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
            random_state=808,
        ),

        "Condensed KNN": lambda: CondensedKNNClassifier(
            k=k,
            p=p,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
            random_state=808,
        ),

    }

    for model_name, factory in models.items():

        result = five_by_two_cv(
            factory,
            X_eval,
            y_eval,
            classification_error,
            stratify=True,
            seed=808,
            label=model_name,
        )

        results.append(result)

    return results


def run_regression_models(
    dataset_name,
    config,
    X_eval,
    y_eval,
    params,
    epsilon,
):
    """
    Evaluate regression models using tuned k, p, gamma, and epsilon.
    """

    results = []

    k = int(params["k"])
    p = float(params["p"])
    gamma = float(params["gamma"])

    models = {
        "Null": lambda: NullRegressor(),

        "KNN": lambda: KNNRegressor(
            k=k,
            p=p,
            gamma=gamma,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
        ),

        "Edited KNN": lambda: EditedKNNRegressor(
            k=k,
            p=p,
            gamma=gamma,
            epsilon=epsilon,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
        ),

        "Condensed KNN": lambda: CondensedKNNRegressor(
            k=k,
            p=p,
            gamma=gamma,
            epsilon=epsilon,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            cyclic_features=config.cyclic_features,
        ),
    }

    for model_name, factory in models.items():

        result = five_by_two_cv(
            factory,
            X_eval,
            y_eval,
            mean_squared_error,
            stratify=False,
            seed=808,
            label=model_name,
        )

        results.append(result)

    return results


# ============================================================
# Result conversion
# ============================================================

def results_to_rows(
    dataset_name,
    results,
    tuned_params=None,
    epsilon=None,
):
    """
    Convert CVResult objects into report-friendly rows.
    """

    rows = []

    for result in results:

        row = {
            "dataset": dataset_name,
            "model": result.label,
            "metric": result.metric_name,
            "mean": result.mean,
            "std": result.std,
            "min": min(result.scores),
            "max": max(result.scores),
        }

        if tuned_params is not None:
            row["k"] = tuned_params.get("k")
            row["p"] = tuned_params.get("p")
            row["gamma"] = tuned_params.get("gamma")

        if epsilon is not None:
            row["epsilon"] = epsilon

        rows.append(row)

    return rows


# ============================================================
# Fold-level results
# ============================================================

def fold_rows(dataset_name, results):
    """
    Preserve all ten individual CV scores.

    This is useful for statistical tests and plots.
    """

    rows = []

    for result in results:

        for fold_number, score in enumerate(result.scores, start=1):

            rows.append({
                "dataset": dataset_name,
                "model": result.label,
                "fold": fold_number,
                "score": score,
            })

    return rows


# ============================================================
# Main experiment
# ============================================================

def run_dataset(dataset_name):

    print()
    print("=" * 72)
    print(f"DATASET: {dataset_name}")
    print("=" * 72)

    X, y, config = load_dataset(dataset_name)

    print(f"Rows:     {len(X)}")
    print(f"Features: {len(X.columns)}")

    stratify = is_classification(dataset_name)

    # --------------------------------------------------------
    # Holdout split
    # --------------------------------------------------------

    X_tune, y_tune, X_eval, y_eval = holdout_split(
        X,
        y,
        frac=0.20,
        stratify=stratify,
        seed=808,
    )

    print(f"Tuning rows:     {len(X_tune)}")
    print(f"Evaluation rows: {len(X_eval)}")

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if is_classification(dataset_name):

        print("\nTuning classifier...")

        tuning = tune_classifier(
            config,
            X_tune,
            y_tune,
        )

        print("Best parameters:")
        print(tuning.params)

        tuning.table.to_csv(
            RESULTS_DIR / f"{dataset_name}_knn_tuning.csv",
            index=False,
        )

        results = run_classification_models(
            dataset_name,
            config,
            X_eval,
            y_eval,
            tuning.params,
        )

        summary_rows = results_to_rows(
            dataset_name,
            results,
            tuned_params=tuning.params,
        )

        fold_result_rows = fold_rows(
            dataset_name,
            results,
        )

        return summary_rows, fold_result_rows

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    print("\nTuning regressor...")

    tuning = tune_regressor(
        config,
        X_tune,
        y_tune,
    )

    print("Best parameters:")
    print(tuning.params)

    tuning.table.to_csv(
        RESULTS_DIR / f"{dataset_name}_knn_tuning.csv",
        index=False,
    )

    print("\nTuning epsilon...")

    epsilon, epsilon_table = tune_epsilon(
        dataset_name,
        config,
        X_tune,
        y_tune,
    )

    print(f"Best epsilon: {epsilon}")

    epsilon_table.to_csv(
        RESULTS_DIR / f"{dataset_name}_epsilon_tuning.csv",
        index=False,
    )

    results = run_regression_models(
        dataset_name,
        config,
        X_eval,
        y_eval,
        tuning.params,
        epsilon,
    )

    summary_rows = results_to_rows(
        dataset_name,
        results,
        tuned_params=tuning.params,
        epsilon=epsilon,
    )

    fold_result_rows = fold_rows(
        dataset_name,
        results,
    )

    return summary_rows, fold_result_rows


# ============================================================
# Run everything
# ============================================================

def run_all_experiments():

    make_results_dir()

    summary_rows = []
    fold_result_rows = []

    for dataset_name in DATASET_NAMES:

        summary, folds = run_dataset(dataset_name)

        summary_rows.extend(summary)
        fold_result_rows.extend(folds)

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_result_rows)

    # --------------------------------------------------------
    # Save machine-readable results
    # --------------------------------------------------------

    summary_df.to_csv(
        RESULTS_DIR / "all_results.csv",
        index=False,
    )

    folds_df.to_csv(
        RESULTS_DIR / "all_fold_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report-friendly tables
    # --------------------------------------------------------

    report_table = summary_df[
        [
            "dataset",
            "model",
            "mean",
            "std",
        ]
    ].copy()

    report_table["mean_std"] = (
        report_table["mean"].map(lambda x: f"{x:.4f}")
        + " ± "
        + report_table["std"].map(lambda x: f"{x:.4f}")
    )

    report_table.to_csv(
        RESULTS_DIR / "report_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # LaTeX table
    # --------------------------------------------------------

    latex_table = report_table[
        [
            "dataset",
            "model",
            "mean_std",
        ]
    ].to_latex(
        index=False,
        escape=True,
    )

    with open(
        RESULTS_DIR / "report_summary.tex",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(latex_table)

    print()
    print("=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)

    print("\nSummary:")
    print(
        report_table[
            ["dataset", "model", "mean_std"]
        ].to_string(index=False)
    )

    print("\nFiles written to:")
    print(RESULTS_DIR.resolve())


if __name__ == "__main__":
    run_all_experiments()