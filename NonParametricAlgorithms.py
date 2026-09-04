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
import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.min_values = {}
        self.max_values = {}

    def fit(self, X):#preprocessor.fit(X_train), training
        for col in self.numeric_cols:
            self.min_values[col] = X[col].min()
            self.max_values[col] = X[col].max()

    def transform(self, X):#X_test = preprocessor.transform(X_test), testing
        X_prime = X.copy()

        for col in self.numeric_cols:
            min = self.min_values[col]
            max = self.max_values[col]

            if max != min:
                X_prime[col] = (X_prime[col] - min) / (max - min)
            else:
                X_prime[col] = 0

        return X_prime

    def fit_transform(self, X):# X_train = preprocessor.fit_transform(X_train)
        self.fit(X)
        return self.transform(X)


columns = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Visceral weight',
    'Shell weight',
    'Rings'
]

df = pd.read_csv('datasets/abalone.data',names=columns)

#X is features, y is target
D = df.drop('Rings',axis=1)
g = df['Rings']

D_train, D_test, g_train, g_test = train_test_split(D, g, test_size=0.6, random_state=808)



num_cols = [
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Visceral weight',
    'Shell weight',
]

cat_cols = ['Sex']

preprocessor = Preprocessor(num_cols, cat_cols)

D_train_processed = preprocessor.fit_transform(D_train)
D_test_processed = preprocessor.fit_transform(D_test)

#so now data is preprocessed with numeric data falling between 0-1
print(D_train_processed.head())




columns = [
    'buying',
    'maint',
    'doors',
    'persons',
    'lug_boot',
    'safety',
    'class'
]

df = pd.read_csv('datasets/car.data',names=columns)

X = df.drop('class',axis=1)
y = df['class']

