import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        regularization=0.01,
        max_iter=1000,
        tol=1e-4,
        early_stopping=True,
        patience=10,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.scaler = StandardScaler()

    def sigmoid(self, z):
        return expit(z)

    def compute_cost(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))

        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)

        cost = -1 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) + (
            self.regularization / (2 * m)
        ) * np.sum(np.square(weights[1:]))

        return cost

    def compute_gradient(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))

        error = h - y
        gradient = (1 / m) * np.dot(X.T, error)

        regularization_term = (self.regularization / m) * weights
        regularization_term[0] = 0
        gradient += regularization_term

        return gradient

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = self.scaler.fit_transform(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.zeros(X.shape[1])
        best_cost = np.inf
        no_improve = 0

        for i in range(self.max_iter):
            gradient = self.compute_gradient(X, y, self.weights)

            # Update weights
            self.weights -= self.learning_rate * gradient
            cost = self.compute_cost(X, y, self.weights)

            if self.verbose:
                print(f"Iteration {i + 1}: Cost {cost:.4f}")

            if not self.early_stopping:
                continue
            if cost < best_cost - self.tol:
                best_cost = cost
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
