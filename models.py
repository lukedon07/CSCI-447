#nearest neighbor methods work best with numeric features
#first normalize categorical features using min-max or z-score normalization
#One way of handling categorical features is with the Value Difference Metric (VDM). You are
#not required to use VDM, but you may find this to be a useful method
#VDM only works with classification. If you have categorical features in regression,
#you will need to apply a different approach
#(such as one hot coding with Hamming distance, which would also work for regression)


#when finding nearest neighbors use Minkowski's metric of the form


#First attempt at required steps before starting, open to changes
#Step One: Import data, identify numerical/categorical columns
#Step Two: Preprocess, (min max normalization for numeric) (one hot or label encoding for categorical)
#Step Three: Create a general distance function
#Step Four: Create two KNN functions (classification,regression)
#Classifier, car, votes, cancer
#Regressor, abalone, fire, machine


import numpy as np
from collections import Counter

from preprocessing import Preprocessor



class NullClassifier:
    #Predicts by selecting the most common class
    def __init__(self):
        self.prediction = None

    def fit(self,X,y):
        y = np.asarray(y)

        if len(y) == 0:
            raise ValueError("Empty, cannot train")

        self.prediction = Counter(y).most_common(1)[0][0]

        return self

    def predict(self,X):
        if self.prediction is None:
            raise RuntimeError("Model must be fitted before prediction")
        return np.full(len(X), self.prediction)


class NullRegressor:
    #Predicts the average value observed
    def __init__(self):
        self.predction = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("Empty, cannot train")

        self.predction = float(np.mean(y))
        return self

    def predict(self,X):
        if self.predction is None:
            raise RuntimeError("Model must be fitted before prediction")

        return np.full(len(X), self.predction)