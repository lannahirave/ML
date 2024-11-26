import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from logger import get_logger


class MyLinearRegression:
    def __init__(
        self,
        weights_init="random",
        add_bias=True,
        learning_rate=1e-4,
        num_iterations=1_000,
        verbose=False,
        max_error=1e-5,
    ):
        """Linear regression model using gradient descent

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int
                maximum number of iterations in gradient descent
            verbose: bool
                enabling verbose output
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        """

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error

        if self.verbose:
            self.logger = get_logger("lab1/part1/logs/part1.log")

    def initialize_weights(self, n_features):
        """weights initialization function"""
        if self.weights_init == "random":
            weights = np.random.rand(
                n_features, 1
            )  # Initialize weights with random values from a normal distribution

        elif self.weights_init == "zeros":
            weights = np.zeros((n_features, 1))  # Initialize weights with zeros
        else:
            raise NotImplementedError
        return weights

    def cost(self, target, pred):
        """calculate cost function

        # Arguments:
            target: np.array
                array of target floating point numbers
            pred: np.array
                array of predicted floating points numbers
        """

        loss = np.mean((pred - target) ** 2)

        return loss

    def fit(self, x, y):
        """Train the model using gradient descent"""
        if self.add_bias:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        n_features = x.shape[1]
        self.weights = self.initialize_weights(n_features)

        for i in range(self.num_iterations):
            predictions = x @ self.weights
            current_loss = self.cost(y, predictions)

            gradient = -2 / x.shape[0] * x.T @ (y - predictions)
            self.weights -= self.learning_rate * gradient

            new_predictions = x @ self.weights
            new_loss = self.cost(y, new_predictions)

            if self.verbose and i % 1000 == 0:
                self.logger.Log(f"Iteration {i+1}: Loss = {new_loss}")

            if abs(new_loss - current_loss) < self.max_error:
                if self.verbose:
                    self.logger.Log(f"Converged at iteration {i+1}.")
                break

    def predict(self, x):
        """Predict values for new data"""
        if self.add_bias:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = x @ self.weights
        return y_hat


def normal_equation(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


if __name__ == "__main__":
    folder_img = "lab1/part1/plots/"
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100, 1)

    # normalization of input data
    x /= np.max(x)

    plt.title("Data samples")
    plt.scatter(x, y)
    plt.savefig(folder_img + "data_samples.png")

    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title("Data samples with sklearn model")
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color="r")
    plt.savefig(folder_img + "sklearn_model.png")
    print("Sklearn MSE: ", mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression(verbose=True, num_iterations=1000000)
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title("Data samples with my model")
    plt.scatter(x, y)
    plt.plot(x, y_hat, color="r")
    plt.savefig(folder_img + "my_model.png")
    print("My MSE: ", mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title("Data samples with normal equation")
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color="r")
    plt.savefig(folder_img + "normal_equation.png")
    print("Normal equation MSE: ", mean_squared_error(y, y_hat_normal))
