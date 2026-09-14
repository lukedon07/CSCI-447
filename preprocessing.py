import numpy as np
import pandas as pd


class Preprocessor:
    """
    Dataset-aware preprocessing.

    Numeric features:
        Min-max normalization using TRAINING data only.

    Categorical features:
        Left unchanged.

    Cyclic features:
        Left unchanged. DistanceCalculator handles their cyclic geometry.
    """

    def __init__(
        self,
        numeric_cols,
        categorical_cols=None,
        cyclic_cols=None,
    ):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.cyclic_cols = dict(cyclic_cols or {})

        self.min_values = {}
        self.max_values = {}

        self.fitted = False

    def fit(self, X):
        X = X.copy()

        missing = [
            col for col in self.numeric_cols
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing numeric columns during preprocessing: {missing}"
            )

        for col in self.numeric_cols:
            values = pd.to_numeric(X[col], errors="raise")

            self.min_values[col] = float(values.min())
            self.max_values[col] = float(values.max())

        self.fitted = True

        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()."
            )

        X = X.copy()

        for col in self.numeric_cols:
            if col not in X.columns:
                raise ValueError(
                    f"Missing numeric column during transform: '{col}'"
                )

            min_val = self.min_values[col]
            max_val = self.max_values[col]

            values = pd.to_numeric(X[col], errors="raise")

            if max_val != min_val:
                X[col] = (
                    (values - min_val)
                    / (max_val - min_val)
                )
            else:
                # Constant training feature carries no distance information.
                X[col] = 0.0

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class DistanceCalculator:
    """
    Distance calculator for mixed numeric, categorical, and cyclic data.

    Numeric:
        Minkowski distance after normalization.

    Categorical:
        Hamming-style distance:
            0 if values match
            1 if values differ

    Cyclic:
        Circular distance normalized by the feature period.

    The classifier and regressor both use this same distance definition.
    """

    def __init__(
        self,
        numeric_cols,
        categorical_cols=None,
        cyclic_cols=None,
        p=2,
    ):
        if p <= 0:
            raise ValueError("p must be greater than 0")

        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.cyclic_cols = dict(cyclic_cols or {})
        self.p = float(p)

        for col, period in self.cyclic_cols.items():
            if period <= 0:
                raise ValueError(
                    f"Cyclic period for '{col}' must be greater than 0"
                )

    def numeric_distance(self, x, y):
        if not self.numeric_cols:
            return 0.0

        powered_sum = 0.0

        for col in self.numeric_cols:
            difference = abs(
                float(x[col]) - float(y[col])
            )

            powered_sum += difference ** self.p

        return powered_sum ** (1.0 / self.p)

    def categorical_distance(self, x, y):
        if not self.categorical_cols:
            return 0.0

        distance = 0.0

        for col in self.categorical_cols:
            if x[col] != y[col]:
                distance += 1.0

        return distance

    def cyclic_distance(self, x, y):
        if not self.cyclic_cols:
            return 0.0

        powered_sum = 0.0

        for col, period in self.cyclic_cols.items():
            difference = abs(
                float(x[col]) - float(y[col])
            )

            circular_difference = min(
                difference,
                period - difference,
            )

            normalized_difference = (
                circular_difference / period
            )

            powered_sum += normalized_difference ** self.p

        return powered_sum ** (1.0 / self.p)

    def distance(self, x, y):
        numeric = self.numeric_distance(x, y)
        categorical = self.categorical_distance(x, y)
        cyclic = self.cyclic_distance(x, y)

        return numeric + categorical + cyclic

    # ------------------------------------------------------------------
    # vectorized version
    # ------------------------------------------------------------------
    #
    # the methods above are the actual definition of the distance and are
    # easier to read, but way too slow to run the experiments with. finding
    # one point's neighbors means n calls to distance(), and every call does
    # a .iloc lookup that rebuilds a pandas Series. on abalone that's about
    # 2.8 million row lookups per fold, which was going to take ~12 hours for
    # a full run.
    #
    # this does the same math a different way. convert the training set to
    # numpy arrays once, then get all n distances in one array operation
    # instead of n loop iterations. same Minkowski sum, same Hamming count,
    # same circular distance, so the neighbors and predictions come out
    # identical. checked that against the old version before switching.
    #
    #     calc.fit_reference(X_train)          # once in fit()
    #     d = calc.distances_to_reference(x)   # per query point
    #
    # categorical values get turned into int codes so comparing is just int
    # equality. codes come from the training set, and anything not seen in
    # training gets -1, which can't match a real code, so it counts as
    # different from everything. same as what the row version does.

    def fit_reference(self, X):
        """set up the training matrix so repeated queries are fast."""
        X = X.reset_index(drop=True)

        self._ref_n = len(X)
        self._ref_num = self._numeric_matrix(X)
        self._cat_codes = [
            {value: code for code, value in enumerate(pd.unique(X[col]))}
            for col in self.categorical_cols
        ]
        self._ref_cat = self._categorical_matrix(X)
        self._ref_cyc = self._cyclic_matrix(X)

        return self

    def _numeric_matrix(self, X):
        if not self.numeric_cols:
            return np.zeros((len(X), 0), dtype=float)
        return X[self.numeric_cols].to_numpy(dtype=float)

    def _categorical_matrix(self, X):
        if not self.categorical_cols:
            return np.zeros((len(X), 0), dtype=np.int64)

        columns = []
        for col, codes in zip(self.categorical_cols, self._cat_codes):
            # -1 for anything not in training, it matches nothing
            columns.append(X[col].map(codes).fillna(-1).to_numpy(dtype=np.int64))

        return np.column_stack(columns)

    def _cyclic_matrix(self, X):
        if not self.cyclic_cols:
            return np.zeros((len(X), 0), dtype=float)
        return X[list(self.cyclic_cols.keys())].to_numpy(dtype=float)

    def _row_vectors(self, x):
        """break one row out into the same three array pieces."""
        num = (np.array([float(x[c]) for c in self.numeric_cols], dtype=float)
               if self.numeric_cols else np.zeros(0, dtype=float))

        if self.categorical_cols:
            cat = np.array(
                [codes.get(x[col], -1)
                 for col, codes in zip(self.categorical_cols, self._cat_codes)],
                dtype=np.int64,
            )
        else:
            cat = np.zeros(0, dtype=np.int64)

        cyc = (np.array([float(x[c]) for c in self.cyclic_cols], dtype=float)
               if self.cyclic_cols else np.zeros(0, dtype=float))

        return num, cat, cyc

    def _distances(self, num, cat, cyc):
        """distance from one prepared point to every training row."""
        total = np.zeros(self._ref_n, dtype=float)

        # Minkowski on the numeric columns, every row at once
        if self.numeric_cols:
            diffs = np.abs(self._ref_num - num)
            total += (diffs ** self.p).sum(axis=1) ** (1.0 / self.p)

        # Hamming, just count how many columns don't match per row
        if self.categorical_cols:
            total += (self._ref_cat != cat).sum(axis=1).astype(float)

        # circular distance, scaled by each feature's period
        if self.cyclic_cols:
            periods = np.array(list(self.cyclic_cols.values()), dtype=float)
            diffs = np.abs(self._ref_cyc - cyc)
            circular = np.minimum(diffs, periods - diffs)
            normalized = circular / periods
            total += (normalized ** self.p).sum(axis=1) ** (1.0 / self.p)

        return total

    def distances_to_reference(self, x):
        """distances from query row x to every training row."""
        if not hasattr(self, "_ref_n"):
            raise RuntimeError("fit_reference() must be called first")
        return self._distances(*self._row_vectors(x))

    def distances_from_reference_row(self, i):
        """
        distances from training row i to every training row.

        this is for the edited variants, since their fit compares every point
        against every other one. pulling row i straight out of the arrays
        means not going back through pandas n times.
        """
        if not hasattr(self, "_ref_n"):
            raise RuntimeError("fit_reference() must be called first")
        return self._distances(
            self._ref_num[i] if self.numeric_cols else np.zeros(0),
            self._ref_cat[i] if self.categorical_cols else np.zeros(0, dtype=np.int64),
            self._ref_cyc[i] if self.cyclic_cols else np.zeros(0),
        )
