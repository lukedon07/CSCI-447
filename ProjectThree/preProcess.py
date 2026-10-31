from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _read(stem, **kwargs):
    """Read datasets/<stem>.data (or .csv). Extra files such as .names are ignored."""
    matches = [p for p in sorted(DATA_DIR.glob(stem + ".*")) if p.suffix in (".data", ".csv")]
    if not matches:
        raise FileNotFoundError(f"No '{stem}.data' or '{stem}.csv' found in {DATA_DIR}")
    return pd.read_csv(matches[0], skipinitialspace=True, **kwargs)


def one_hot(values, categories):
    """One-hot encode `values` using a fixed list of categories (n, len(categories))."""
    values = list(values)
    unknown = set(values) - set(categories)
    if unknown:
        raise ValueError(f"Unexpected values {sorted(map(str, unknown))}; expected {list(categories)}")
    index = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(values), len(categories)))
    out[np.arange(len(values)), [index[v] for v in values]] = 1.0
    return out


def ordinal(values, order):
    """Map each value to its position in `order` (an ordered list of categories)."""
    values = list(values)
    unknown = set(values) - set(order)
    if unknown:
        raise ValueError(f"Unexpected values {sorted(map(str, unknown))}; expected {list(order)}")
    return np.array([order.index(v) for v in values], dtype=float)


# --------------------------------------------------------------------------
# Dataset loaders: each returns (X, y)
# --------------------------------------------------------------------------
def load_breast_cancer():
    """9 numeric features. ID column dropped. '?' -> NaN. Class 2 -> 0, 4 -> 1."""
    df = _read("breast-cancer-wisconsin", header=None, na_values="?")
    labels = df.iloc[:, 10]
    if not set(labels.unique()) <= {2, 4}:
        raise ValueError("Breast cancer class column should only contain 2 and 4")
    X = df.iloc[:, 1:10].to_numpy(dtype=float)
    y = (labels == 4).to_numpy(dtype=int)
    return X, y


CAR_ORDERS = [
    ["low", "med", "high", "vhigh"],  # buying
    ["low", "med", "high", "vhigh"],  # maint
    ["2", "3", "4", "5more"],  # doors
    ["2", "4", "more"],  # persons
    ["small", "med", "big"],  # lug_boot
    ["low", "med", "high"],  # safety
]
CAR_CLASSES = ["unacc", "acc", "good", "vgood"]


def load_car():
    """6 ordinal attributes (coded by their natural order). 4 classes -> 0..3."""
    df = _read("car", header=None, dtype=str)
    X = np.column_stack([ordinal(df.iloc[:, i], CAR_ORDERS[i]) for i in range(6)])
    y = ordinal(df.iloc[:, 6], CAR_CLASSES).astype(int)
    return X, y


def load_votes():
    """16 votes, each one-hot over {y, n, ?} (the '?' is kept as its own category).
    democrat -> 0, republican -> 1."""
    df = _read("house-votes-84", header=None, dtype=str, keep_default_na=False)
    y = ordinal(df.iloc[:, 0], ["democrat", "republican"]).astype(int)
    X = np.hstack([one_hot(df.iloc[:, i], ["y", "n", "?"]) for i in range(1, 17)])
    return X, y


def load_abalone():
    """Sex one-hot (M, F, I) + 7 numeric measurements. Target: Rings."""
    df = _read("abalone", header=None)
    sex = one_hot(df.iloc[:, 0], ["M", "F", "I"])
    numeric = df.iloc[:, 1:8].to_numpy(dtype=float)
    return np.hstack([numeric, sex]), df.iloc[:, 8].to_numpy(dtype=float)


def load_hardware():
    """6 numeric features. Vendor, model name and ERP (the authors' own
    estimate of the target) are dropped. Target: PRP."""
    df = _read("machine", header=None)
    X = df.iloc[:, 2:8].to_numpy(dtype=float)
    y = df.iloc[:, 8].to_numpy(dtype=float)
    return X, y


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def load_forest_fires():
    """X, Y + 8 weather columns + one-hot month + one-hot day.
    Target: log(area + 1), because burned area is extremely skewed."""
    df = _read("forestfires")
    numeric = df.drop(columns=["month", "day", "area"]).to_numpy(dtype=float)
    month = one_hot(df["month"].str.lower(), MONTHS)
    day = one_hot(df["day"].str.lower(), DAYS)
    y = np.log1p(df["area"].to_numpy(dtype=float))
    return np.hstack([numeric, month, day]), y


# name -> (loader, task, number of classes)
DATASETS = {
    "breast_cancer": (load_breast_cancer, "classification", 2),
    "car": (load_car, "classification", 4),
    "votes": (load_votes, "classification", 2),
    "abalone": (load_abalone, "regression", None),
    "hardware": (load_hardware, "regression", None),
    "forest_fires": (load_forest_fires, "regression", None),
}


def load_dataset(name):
    """Return (X, y, task, n_classes). X may still contain NaN."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}")
    loader, task, n_classes = DATASETS[name]
    X, y = loader()
    return X, y, task, n_classes


# --------------------------------------------------------------------------
# Fold preprocessing (everything is learned from the training fold only)
# --------------------------------------------------------------------------
def preprocess_fold(X_train, X_test, y_train, y_test, task, n_classes=None):
    """Impute, standardize and encode targets.

    Returns X_train, X_test, y_train, y_test, y_stats where
      * features are median-imputed and z-scored with training statistics,
      * classification targets are one-hot (n, n_classes); y_stats is None,
      * regression targets are z-scored columns (n, 1); y_stats = (mean, std),
        so predictions can be mapped back with  y_pred * std + mean.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # 1. Missing values -> training median (0 if a column is entirely missing)
    medians = pd.DataFrame(X_train).median().fillna(0.0).to_numpy()
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    # 2. Z-score features (constant columns get std 1 so they become 0, not NaN)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 3. Targets
    if task == "classification":
        eye = np.eye(n_classes)
        return X_train, X_test, eye[np.asarray(y_train, dtype=int)], eye[np.asarray(y_test, dtype=int)], None

    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1, 1)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    return X_train, X_test, (y_train - y_mean) / y_std, (y_test - y_mean) / y_std, (y_mean, y_std)


if __name__ == "__main__":
    for name in DATASETS:
        X, y, task, k = load_dataset(name)
        print(f"{name:14s} {task:14s} X={X.shape}  NaN={int(np.isnan(X).sum())}")
