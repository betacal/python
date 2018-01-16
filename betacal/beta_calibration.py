from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


import warnings


def _beta_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]

    m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    map = [a, b, m]
    return map, lr


class _BetaCal(BaseEstimator, RegressorMixin):
    """Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]


def _beta_am_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.log(df / (1. - df))

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]
    inter = lr.intercept_[0]
    a = coefs[0]
    b = a
    m = 1.0 / (1.0 + np.exp(inter / a))
    map = [a, b, m]
    return map, lr


class _BetaAMCal(BaseEstimator, RegressorMixin):
    """Beta regression model with two parameters (a and m, fixing a = b)
    introduced in Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration:
    a well-founded and easily implemented improvement on logistic calibration
    for binary classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m], where a = b

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_am_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.log(df / (1. - df))
        return self.lr_.predict_proba(x)[:, 1]


def _beta_ab_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(2 * x)

    lr = LogisticRegression(fit_intercept=False, C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]
    a = coefs[0]
    b = -coefs[1]
    m = 0.5
    map = [a, b, m]
    return map, lr


class _BetaABCal(BaseEstimator, RegressorMixin):
    """Beta regression model with two parameters (a and b, fixing m = 0.5)
    introduced in Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration:
    a well-founded and easily implemented improvement on logistic calibration
    for binary classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m], where m = 0.5

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_ab_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.hstack((df, 1. - df))
        x = np.log(2 * x)
        return self.lr_.predict_proba(x)[:, 1]


def _beta_a_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.log(df / (1. - df))

    lr = LogisticRegression(fit_intercept=False, C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]
    a = coefs[0]
    b = a
    m = 0.5
    map = [a, b, m]
    return map, lr


class _BetaACal(BaseEstimator, RegressorMixin):
    """Beta regression model with one parameter (a = b, fixing m = 0.5)
    introduced in Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration:
    a well-founded and easily implemented improvement on logistic calibration
    for binary classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m], where a = b and m = 0.5

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_a_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1-eps)

        x = np.log(df / (1. - df))
        return self.lr_.predict_proba(x)[:, 1]
