import numpy as np
import pandas as pd
from collections import Counter

from preprocessing import Preprocessor, DistanceCalculator


class KNNClassifier:
    """
    k-nearest-neighbor classifier.

    Numeric features use Minkowski distance after min-max normalization.
    Categorical features use Hamming distance.
    Cyclic features use circular distance.

    Parameters
    ----------
    k : int
        Number of neighbors.
    p : float
        Minkowski exponent.
    numeric_features : list[str]
        Features to treat as numeric.
    categorical_features : list[str]
        Features to treat as categorical.
    cyclic_features : dict[str, int]
        Cyclic features and their periods.
    random_state : int
        Seed used to break classification ties reproducibly.
    """

    def __init__(
        self,
        k=5,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
        random_state=808,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        if p <= 0:
            raise ValueError("p must be greater than 0")

        self.k = int(k)
        self.p = p
        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})
        self.random_state = random_state

        self.X_train = None
        self.y_train = None
        self.preprocessor = None
        self.distance_calculator = None

        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")

        if self.k > len(X):
            raise ValueError(
                f"k={self.k} is larger than the training set size ({len(X)})"
            )

        expected_features = (
            self.numeric_features
            + self.categorical_features
            + list(self.cyclic_features.keys())
        )

        missing = [col for col in expected_features if col not in X.columns]

        if missing:
            raise ValueError(
                f"Dataset is missing configured features: {missing}"
            )

        self.preprocessor = Preprocessor(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
        )

        self.X_train = self.preprocessor.fit_transform(X)
        self.y_train = np.asarray(y)

        self.distance_calculator = DistanceCalculator(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
            p=self.p,
        )

        return self

    def nearest_neighbors(self, x):
        distances = []

        for i in range(len(self.X_train)):
            distance = self.distance_calculator.distance(
                x,
                self.X_train.iloc[i],
            )

            distances.append((distance, i))

        distances.sort(key=lambda pair: pair[0])

        return distances[:self.k]

    def predict(self, X):
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self.preprocessor.transform(
            X.reset_index(drop=True)
        )

        predictions = []

        for _, x in X.iterrows():
            neighbors = self.nearest_neighbors(x)

            labels = [
                self.y_train[index]
                for _, index in neighbors
            ]

            counts = Counter(labels)
            highest_count = max(counts.values())

            winners = [
                label
                for label, count in counts.items()
                if count == highest_count
            ]

            prediction = self.rng.choice(winners)
            predictions.append(prediction)

        return np.asarray(predictions)



class KNNRegressor:
    """
    k-nearest-neighbor regressor using Gaussian/RBF kernel weighting.

    The assignment specifies:

        K(x, x_q) = exp(-gamma * ||x - x_q||_2^2)

    and the prediction is the weighted average of the k nearest training
    examples.

    Numeric features are min-max normalized before distance calculation.
    Categorical features use Hamming distance.
    Cyclic features use circular distance.

    Parameters
    ----------
    k : int
        Number of neighbors.
    gamma : float
        Gaussian kernel bandwidth parameter. Must be positive.
    p : float
        Minkowski exponent used by the underlying distance calculator.
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    cyclic_features : dict[str, int]
        Cyclic feature names mapped to their periods.
    """

    def __init__(
        self,
        k=5,
        gamma=1.0,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        if p <= 0:
            raise ValueError("p must be greater than 0")

        self.k = int(k)
        self.gamma = float(gamma)
        self.p = p

        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})

        self.X_train = None
        self.y_train = None
        self.preprocessor = None
        self.distance_calculator = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")

        if self.k > len(X):
            raise ValueError(
                f"k={self.k} is larger than the training set size ({len(X)})"
            )

        expected_features = (
            self.numeric_features
            + self.categorical_features
            + list(self.cyclic_features.keys())
        )

        missing = [col for col in expected_features if col not in X.columns]

        if missing:
            raise ValueError(
                f"Dataset is missing configured features: {missing}"
            )

        self.preprocessor = Preprocessor(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
        )

        self.X_train = self.preprocessor.fit_transform(X)
        self.y_train = np.asarray(y, dtype=float)

        self.distance_calculator = DistanceCalculator(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
            p=self.p,
        )

        return self

    def nearest_neighbors(self, x):
        distances = []

        for i in range(len(self.X_train)):
            distance = self.distance_calculator.distance(
                x,
                self.X_train.iloc[i],
            )

            distances.append((distance, i))

        distances.sort(key=lambda pair: pair[0])

        return distances[:self.k]

    def predict(self, X):
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self.preprocessor.transform(
            X.reset_index(drop=True)
        )

        predictions = []

        for _, x in X.iterrows():
            neighbors = self.nearest_neighbors(x)

            weighted_sum = 0.0
            weight_total = 0.0

            for distance, index in neighbors:
                weight = np.exp(-self.gamma * (distance ** 2))

                weighted_sum += weight * self.y_train[index]
                weight_total += weight

            if weight_total == 0:
                # Numerically defensive fallback.
                prediction = float(
                    np.mean([
                        self.y_train[index]
                        for _, index in neighbors
                    ])
                )
            else:
                prediction = weighted_sum / weight_total

            predictions.append(prediction)

        return np.asarray(predictions)



class EditedKNNClassifier:
    """
    Edited k-nearest-neighbor classifier.

    Editing is performed with k=1. A training example is removed when its
    nearest neighbor has a different class.

    After editing, prediction uses the user-specified k.
    """

    def __init__(
        self,
        k=5,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
        random_state=808,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = int(k)
        self.p = p

        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})

        self.random_state = random_state

        self.X_train = None
        self.y_train = None

        self.X_reduced = None
        self.y_reduced = None

        self.preprocessor = None
        self.distance_calculator = None

        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.X_train = X.copy()
        self.y_train = np.asarray(y)

        self.preprocessor = Preprocessor(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
        )

        X_processed = self.preprocessor.fit_transform(X)

        self.distance_calculator = DistanceCalculator(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
            p=self.p,
        )

        keep_indices = []

        for i in range(len(X_processed)):
            best_distance = float("inf")
            nearest_index = None

            for j in range(len(X_processed)):
                if i == j:
                    continue

                distance = self.distance_calculator.distance(
                    X_processed.iloc[i],
                    X_processed.iloc[j],
                )

                if distance < best_distance:
                    best_distance = distance
                    nearest_index = j

            if nearest_index is None:
                keep_indices.append(i)
                continue

            if self.y_train[nearest_index] == self.y_train[i]:
                keep_indices.append(i)

        # Never allow editing to produce an empty training set.
        if len(keep_indices) == 0:
            keep_indices = [0]

        self.X_reduced = X.iloc[keep_indices].reset_index(drop=True)
        self.y_reduced = self.y_train[keep_indices]

        return self

    def predict(self, X):
        if self.X_reduced is None:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        predictor = KNNClassifier(
            k=min(self.k, len(self.X_reduced)),
            p=self.p,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            cyclic_features=self.cyclic_features,
            random_state=self.random_state,
        )

        predictor.fit(
            self.X_reduced,
            pd.Series(self.y_reduced),
        )

        return predictor.predict(X)

    @property
    def reduction_ratio(self):
        if self.X_train is None or self.X_reduced is None:
            return None

        return 1.0 - (
            len(self.X_reduced) / len(self.X_train))



class EditedKNNRegressor:
    """
    Edited k-nearest-neighbor regressor.

    Editing uses k=1 and an epsilon error threshold.

    A training example is retained when the prediction from its nearest
    other training example is within epsilon of its actual target.

    Prediction after editing uses the tuned k and Gaussian kernel weighting.
    """

    def __init__(
        self,
        k=5,
        gamma=1.0,
        epsilon=1.0,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if p <= 0:
            raise ValueError("p must be greater than 0")

        self.k = int(k)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.p = p

        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})

        self.X_train = None
        self.y_train = None

        self.X_reduced = None
        self.y_reduced = None

        self.preprocessor = None
        self.distance_calculator = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.X_train = X.copy()
        self.y_train = np.asarray(y, dtype=float)

        self.preprocessor = Preprocessor(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
        )

        X_processed = self.preprocessor.fit_transform(X)

        self.distance_calculator = DistanceCalculator(
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
            cyclic_cols=self.cyclic_features,
            p=self.p,
        )

        keep_indices = []

        for i in range(len(X_processed)):
            best_distance = float("inf")
            nearest_index = None

            for j in range(len(X_processed)):
                if i == j:
                    continue

                distance = self.distance_calculator.distance(
                    X_processed.iloc[i],
                    X_processed.iloc[j],
                )

                if distance < best_distance:
                    best_distance = distance
                    nearest_index = j

            if nearest_index is None:
                keep_indices.append(i)
                continue

            prediction = self.y_train[nearest_index]

            error = abs(
                prediction - self.y_train[i]
            )

            if error <= self.epsilon:
                keep_indices.append(i)

        if len(keep_indices) == 0:
            keep_indices = [0]

        self.X_reduced = X.iloc[keep_indices].reset_index(drop=True)
        self.y_reduced = self.y_train[keep_indices]

        return self

    def predict(self, X):
        if self.X_reduced is None:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        predictor = KNNRegressor(
            k=min(self.k, len(self.X_reduced)),
            gamma=self.gamma,
            p=self.p,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            cyclic_features=self.cyclic_features,
        )

        predictor.fit(
            self.X_reduced,
            pd.Series(self.y_reduced),
        )

        return predictor.predict(X)

    @property
    def reduction_ratio(self):
        if self.X_train is None or self.X_reduced is None:
            return None

        return 1.0 - (
            len(self.X_reduced) / len(self.X_train)
        )



class CondensedKNNClassifier:
    """
    Condensed k-nearest-neighbor classifier.

    Condensing uses 1-NN. Misclassified examples are added to the condensed
    training set until another complete pass adds no new examples.

    Prediction uses the tuned k.
    """

    def __init__(
        self,
        k=5,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
        random_state=808,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = int(k)
        self.p = p

        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})

        self.random_state = random_state

        self.X_train = None
        self.y_train = None

        self.X_reduced = None
        self.y_reduced = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.X_train = X.copy()
        self.y_train = np.asarray(y)

        # Start with one example from each class.
        selected = []

        for label in pd.unique(self.y_train):
            first_index = np.flatnonzero(
                self.y_train == label
            )[0]

            selected.append(first_index)

        selected = list(dict.fromkeys(selected))

        changed = True

        while changed:
            changed = False

            remaining = [
                i for i in range(len(X))
                if i not in selected
            ]

            if not remaining:
                break

            predictor = KNNClassifier(
                k=1,
                p=self.p,
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                cyclic_features=self.cyclic_features,
                random_state=self.random_state,
            )

            predictor.fit(
                X.iloc[selected].reset_index(drop=True),
                y.iloc[selected].reset_index(drop=True),
            )

            predictions = predictor.predict(
                X.iloc[remaining].reset_index(drop=True)
            )

            for position, original_index in enumerate(remaining):
                if predictions[position] != self.y_train[original_index]:
                    selected.append(original_index)
                    changed = True

        self.X_reduced = X.iloc[selected].reset_index(drop=True)
        self.y_reduced = self.y_train[selected]

        return self

    def predict(self, X):
        if self.X_reduced is None:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        predictor = KNNClassifier(
            k=min(self.k, len(self.X_reduced)),
            p=self.p,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            cyclic_features=self.cyclic_features,
            random_state=self.random_state,
        )

        predictor.fit(
            self.X_reduced,
            pd.Series(self.y_reduced),
        )

        return predictor.predict(X)

    @property
    def reduction_ratio(self):
        if self.X_train is None or self.X_reduced is None:
            return None

        return 1.0 - (
            len(self.X_reduced) / len(self.X_train)
        )




class CondensedKNNRegressor:
    """
    Condensed k-nearest-neighbor regressor.

    Condensing uses 1-NN and an epsilon error threshold.
    Prediction uses tuned k and gamma.
    """

    def __init__(
        self,
        k=5,
        gamma=1.0,
        epsilon=1.0,
        p=2,
        numeric_features=None,
        categorical_features=None,
        cyclic_features=None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")

        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.k = int(k)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.p = p

        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.cyclic_features = dict(cyclic_features or {})

        self.X_train = None
        self.y_train = None

        self.X_reduced = None
        self.y_reduced = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("Cannot train on an empty dataset")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.X_train = X.copy()
        self.y_train = np.asarray(y, dtype=float)

        # Start with one example.
        selected = [0]

        changed = True

        while changed:
            changed = False

            remaining = [
                i for i in range(len(X))
                if i not in selected
            ]

            if not remaining:
                break

            predictor = KNNRegressor(
                k=1,
                gamma=self.gamma,
                p=self.p,
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                cyclic_features=self.cyclic_features,
            )

            predictor.fit(
                X.iloc[selected].reset_index(drop=True),
                y.iloc[selected].reset_index(drop=True),
            )

            predictions = predictor.predict(
                X.iloc[remaining].reset_index(drop=True)
            )

            for position, original_index in enumerate(remaining):
                error = abs(
                    predictions[position]
                    - self.y_train[original_index]
                )

                if error > self.epsilon:
                    selected.append(original_index)
                    changed = True

        self.X_reduced = X.iloc[selected].reset_index(drop=True)
        self.y_reduced = self.y_train[selected]

        return self

    def predict(self, X):
        if self.X_reduced is None:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        predictor = KNNRegressor(
            k=min(self.k, len(self.X_reduced)),
            gamma=self.gamma,
            p=self.p,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            cyclic_features=self.cyclic_features,
        )

        predictor.fit(
            self.X_reduced,
            pd.Series(self.y_reduced),
        )

        return predictor.predict(X)

    @property
    def reduction_ratio(self):
        if self.X_train is None or self.X_reduced is None:
            return None

        return 1.0 - (
            len(self.X_reduced) / len(self.X_train)
        )





class NullClassifier:
    """Predict the plurality class observed during training."""

    def __init__(self):
        self.prediction = None

    def fit(self, X, y):
        y = np.asarray(y)

        if len(y) == 0:
            raise ValueError(
                "Cannot train on an empty dataset."
            )

        self.prediction = Counter(
            y
        ).most_common(1)[0][0]

        return self

    def predict(self, X):
        if self.prediction is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        return np.full(
            len(X),
            self.prediction,
        )


class NullRegressor:
    """
    Regression baseline that always predicts the mean training target.
    """

    def __init__(self):
        self.prediction = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)

        if len(y) == 0:
            raise ValueError("Cannot train on an empty dataset")

        self.prediction = float(np.mean(y))

        return self

    def predict(self, X):
        if self.prediction is None:
            raise RuntimeError(
                "Model must be fitted before prediction"
            )

        return np.full(
            len(X),
            self.prediction,
            dtype=float,
        )

