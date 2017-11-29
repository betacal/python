from .beta_calibration import _BetaCal, _BetaAMCal, _BetaABCal
from sklearn.base import BaseEstimator, RegressorMixin
from .version import __version__


class BetaCalibration(BaseEstimator, RegressorMixin):
    """Wrapper class for the three Beta regression models introduced in 
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded 
    and easily implemented improvement on logistic calibration for binary  
    classifiers. AISTATS 2017.

    Parameters
    ----------
    parameters : string
        Determines which parameters will be calculated by the model. Possible
        values are: "abm" (default), "am" and "ab"

    Attributes
    ----------
    calibrator_ :
        Internal calibrator object. The type depends on the value of parameters.
    """
    def __init__(self, parameters="abm"):
        if parameters == "abm":
            self.calibrator_ = _BetaCal()
        elif parameters == "am":
            self.calibrator_ = _BetaAMCal()
        elif parameters == "ab":
            self.calibrator_ = _BetaABCal()
        else:
            raise ValueError('Unknown parameters', parameters)

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
            Currently, no sample weighting is done by the models.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.calibrator_.fit(X, y, sample_weight)
        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        : array, shape (n_samples,)
            The predicted values.
        """
        return self.calibrator_.predict(S)
