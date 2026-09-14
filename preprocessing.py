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
