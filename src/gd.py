"""
Linear and logistic regression trained with batch gradient descent, written from scratch
with NumPy (the same approach as Project 3 of my Machine Learning course).

Both classes standardize features internally, so one learning rate works across features
measured on different scales, and both keep the cost history so the learning-rate
behavior can be plotted.
"""
import numpy as np


class _Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class LinearRegressionGD:
    """Minimize  (1/2n) * sum (b + Xw - y)^2  +  (l2/2n) * ||w||^2  by gradient descent."""

    def __init__(self, learning_rate=0.1, n_iter=5000, l2=0.0, tol=1e-12):
        self.learning_rate, self.n_iter, self.l2, self.tol = learning_rate, n_iter, l2, tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.scaler_ = _Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        n, d = Z.shape
        w, b = np.zeros(d), y.mean()
        self.cost_history_ = []
        for _ in range(self.n_iter):
            err = Z @ w + b - y
            cost = (err @ err + self.l2 * w @ w) / (2 * n)
            self.cost_history_.append(cost)
            grad_w = (Z.T @ err + self.l2 * w) / n
            grad_b = err.mean()
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if not np.isfinite(cost):
                break  # learning rate too large: diverged
            if len(self.cost_history_) > 1 and abs(self.cost_history_[-2] - cost) < self.tol:
                break
        self.w_, self.b_ = w, b
        return self

    def predict(self, X):
        return self.scaler_.transform(np.asarray(X, float)) @ self.w_ + self.b_

    @property
    def coef_original_scale_(self):
        """Weights and intercept translated back to the unstandardized features."""
        w = self.w_ / self.scaler_.std_
        return w, self.b_ - self.scaler_.mean_ @ w


class LogisticRegressionGD:
    """Minimize mean cross-entropy + (l2/2n)||w||^2 by gradient descent."""

    def __init__(self, learning_rate=0.1, n_iter=5000, l2=1.0, tol=1e-10):
        self.learning_rate, self.n_iter, self.l2, self.tol = learning_rate, n_iter, l2, tol

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.scaler_ = _Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        n, d = Z.shape
        w, b = np.zeros(d), 0.0
        self.cost_history_ = []
        eps = 1e-12
        for _ in range(self.n_iter):
            p = self._sigmoid(Z @ w + b)
            cost = (-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                    + self.l2 * w @ w / (2 * n))
            self.cost_history_.append(cost)
            grad_w = (Z.T @ (p - y) + self.l2 * w) / n
            grad_b = np.mean(p - y)
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if len(self.cost_history_) > 1 and abs(self.cost_history_[-2] - cost) < self.tol:
                break
        self.w_, self.b_ = w, b
        return self

    def predict_proba(self, X):
        return self._sigmoid(self.scaler_.transform(np.asarray(X, float)) @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
