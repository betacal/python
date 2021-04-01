import unittest
from betacal import BetaCalibration
import numpy as np

# from generate_scores_and_y(0.8, 0.4, 40, 42)
s = [0.43155171, 0.81164906, 0.66544901, 0.53177918, 0.15549116, 0.16259166,
     0.36480309, 0.81877874, 0.94360637, 0.73610746, 0.55979712, 0.85662239,
     0.38039252, 0.7806386 , 0.38548359, 0.7468978 , 0.26136641, 0.21377051,
     0.01270956, 0.05394571, 0.51571459, 0.31820521, 0.65717799, 0.25428535,
     0.45378324, 0.62464611, 0.20519146, 0.87777557, 0.11439908, 0.53848995,
     0.64487573, 0.08061064, 0.46484883, 0.40406019, 0.81670188, 0.9357303 ,
     0.40183604, 0.09328503, 0.9462795 , 0.26967112]

y = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0]


def generate_scores_and_y(accuracy=0.8, prior=0.4, n_samples=40, seed=42):
    np.random.seed(seed)
    y = np.random.binomial(1, prior, n_samples)
    predicted_class = (y == 0).astype(int)
    correct = np.random.binomial(1, accuracy, len(y)).astype(bool)
    predicted_class[correct] = y[correct]
    s = (np.random.rand(len(y))+predicted_class)/2
    return s, y

def map_predictions(s, a, b, m):
    c = b * np.log(1. - m) - a * np.log(m)
    p = np.array(s)
    beta = 1 / (1 + 1 / (np.exp(c) * p ** a / (1 - p) ** b))
    return beta


class BetaCalibrationTests(unittest.TestCase):
    def test_betacal(self):
        bc = BetaCalibration(parameters="abm")
        bc.fit(s, y)

        pred = bc.predict(s)

        beta_map = bc.calibrator_.map_

        pred_abm = map_predictions(s, *beta_map)

        np.testing.assert_allclose(pred, pred_abm, rtol=1e-5, atol=1e-5)
        assert(len(bc.calibrator_.lr_.coef_[0]) == 2)

    def test_betacal_am(self):
        bc = BetaCalibration(parameters="am")
        bc.fit(s, y)

        pred = bc.predict(s)

        beta_map = bc.calibrator_.map_

        pred_am = map_predictions(s, *beta_map)

        np.testing.assert_allclose(pred, pred_am, rtol=1e-5, atol=1e-5)
        assert(len(bc.calibrator_.lr_.coef_[0]) == 1)

    def test_betacal_ab(self):
        bc = BetaCalibration(parameters="ab")
        bc.fit(s, y)

        pred = bc.predict(s)

        beta_map = bc.calibrator_.map_

        pred_ab = map_predictions(s, *beta_map)

        np.testing.assert_allclose(pred, pred_ab, rtol=1e-5, atol=1e-5)
        assert(len(bc.calibrator_.lr_.coef_[0]) == 2)

    def test_betacal_a(self):
        bc = BetaCalibration(parameters="a")
        bc.fit(s, y)

        pred = bc.predict(s)

        beta_map = bc.calibrator_.map_

        pred_a = map_predictions(s, *beta_map)
        
        np.testing.assert_allclose(pred, pred_a, rtol=1e-5, atol=1e-5)
        assert(len(bc.calibrator_.lr_.coef_[0]) == 1)
