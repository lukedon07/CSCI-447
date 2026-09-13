import numpy as np
import pandas as pd


class Preprocessor:
    #Numeric Features: min-max normalization [0-1]
    #Categorical Features: keep as is, distance calculator handles them
    #Cyclic Features: keep as is, distance calculator handles them

    def __init__(self, numeric_cols, categorical_cols=None,cyclic_cols=None):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.cyclic_cols = dict(cyclic_cols or [])
        self.min_values = {}
        self.max_values = {}

    def fit(self, X):#preprocessor.fit(X_train), training
        X = X.copy()

        for col in self.numeric_cols:
            self.min_values[col] = X[col].min()
            self.max_values[col] = X[col].max()
        return self

    def transform(self, X):#X_test = preprocessor.transform(X_test), testing
        X = X.copy()

        for col in self.numeric_cols:
            min_val = self.min_values[col]
            max_val = self.max_values[col]

            if max_val != min_val:
                X[col] = (X[col] - min_val) / (max_val - min_val)
            else:
                X[col] = 0.0
        return X

    def fit_transform(self, X):# X_train = preprocessor.fit_transform(X_train)
        self.fit(X)
        return self.transform(X)






class DistanceCalculator:
    #Numeric Features: Minkowski distance
    #Categorical Features: Hamming distance, 0 for match, 1 for not
    #Cyclic Features: Normalized by the period

    def __init__(self, numeric_cols, categorical_cols=None, cyclic_cols=None, p=2):#never set p to zero please
        if p <= 0:
            raise ValueError("p must be greater than 0")
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.cyclic_cols = dict(cyclic_cols or [])
        self.p = p

    def numeric_distance(self, x, y):
        if not self.numeric_cols:
            return 0.0

        totalDistance = 0.0

        for col in self.numeric_cols:
            diff = abs(float(x[col]) - float(y[col]))
            totalDistance += diff ** self.p
        return totalDistance ** (1 / self.p)#take pth root

    def categorical_distance(self, x, y):
        if not self.categorical_cols:
            return 0.0

        totalDistance = 0.0

        for col in self.categorical_cols:
            if x[col] != y[col]:
                totalDistance += 1.0
        return totalDistance

    def cyclic_distance(self, x, y):
        if not self.cyclic_cols:
            return 0.0

        totalDistance = 0.0

        for col, period in self.cyclic_cols.items():
            difference = abs(float(x[col]) - float(y[col]))

            circular_difference = min(difference, period - difference)

            normalized_difference = circular_difference / period
            totalDistance += normalized_difference ** self.p

        return totalDistance ** (1.0 / self.p)

    def distance(self, x, y):
        num = self.numeric_distance(x, y)
        cat = self.categorical_distance(x, y)
        cyc = self.cyclic_distance(x, y)
        return num + cat + cyc
