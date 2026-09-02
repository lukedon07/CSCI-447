#baseline approach
#naïve algorithms that will simply return the plurality (most common) class label
#in classification tasks, and the average of the outputs for regression tasks

#null models do not consider any of the feature values in its decision, you should not expect
#high accuracy nor low error results

#will test whatever process or pipeline you put in place and serve as a baseline for comparison



#Step one: load datasets
#Step two: separate features, (normalize if needed?)
#Step three: split into data into training or testing
#Step four: fit the null model with training set
#Step five: predict test observations
#Step six: using an appropriate performance metric calculate success

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



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

#X is features, y is target
X = df.drop('class',axis=1)
y = df['class']

#print(df.head())
#print(df.shape)
#print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=808)

class NullClassifier:
    def __init__(self):
        self.prediction = None

    def fit(self,y):
        self.prediction = Counter(y).most_common(1)[0][0]

    def predict(self,X):
        return [self.prediction] * len(X)


model = NullClassifier()
model.fit(y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Null Model Accuracy: ",round(accuracy * 100, 2), "%")#appears to be most accurate at around a 50/50 train-test-split


class NullRegresser:
    def __init__(self):
        self.prediction = None

    def fit(self, y):
        self.prediction = sum(y) / len(y)

    def predict(self, X):
        return [self.prediction] * len(X)


