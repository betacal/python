from __future__ import division

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.base import clone


class AdaBoostClassifier(BaseEstimator):
    def __init__(self, base_estimator=None, n_estimators=50):
        self.base_estimator = base_estimator
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.alphas = np.zeros(n_estimators)
        self.estimators = []

    def fit(self, X, y):
        sample_weights = np.ones(np.alen(X)) * (1.0 / np.alen(X))
        y_changed = y
        y_changed[y == 0] = -1
        for iboost in np.arange(self.n_estimators):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y_changed, sample_weights)
            self.estimators.append(estimator)
            predictions = estimator.predict(X)
            incorrect = (predictions != y).astype(float)
            error = (sample_weights * incorrect).sum()
            if error > 0:
                self.alphas[iboost] = 0.5 * np.log((1-error) / error)
                a = self.alphas[iboost]
                modifier = np.exp(-y_changed * a * predictions)
                sample_weights *= modifier
                sample_weights /= sample_weights.sum()
            else:
                self.alphas[iboost] = 1.0
                self.alphas = self.alphas[:(iboost + 1)]
                self.n_estimators = len(self.estimators)
                break
        return self

    def predict_proba(self, X):
        predictions = np.zeros(np.alen(X))
        for iboost in np.arange(self.n_estimators):
            a = self.alphas[iboost]
            predictions += a * self.estimators[iboost].predict(X)
        probas = 1.0 / (1.0 + np.exp(-2*predictions)).reshape(-1, 1)
        probas = np.hstack((1.0 - probas, probas))
        return probas
