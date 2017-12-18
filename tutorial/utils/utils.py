from sklearn.linear_model import LogisticRegression


class MyLogisticRegression(LogisticRegression):
    def fit(self, x, y, **kwargs):
        return super(MyLogisticRegression, self).fit(
                x.reshape(-1, 1), y, **kwargs)

    def predict(self, x, **kwargs):
        return super(MyLogisticRegression, self).predict_proba(
                x.reshape(-1, 1), **kwargs)[:, 1]
