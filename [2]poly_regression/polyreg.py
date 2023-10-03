"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean: np.ndarray = None # shape: (1, degree)
        self.std: np.ndarray = None # shape: (1, degree)

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        n = len(X)
        P = np.empty((n, degree)) # avoids array copying
        P[:, (0,)] = X
        for i in range(1, degree):
            P[:, (i,)] = P[:, (i - 1,)] * X
        return P

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        n = len(X)
        P = self.polyfeatures(X, self.degree)
        
        self.mean = np.mean(P, axis=0, keepdims=True)
        self.std = np.std(P, axis=0, keepdims=True)
        self.std[self.std == 0] = 1 # avoid division by zero, just in case
        P = (P - self.mean) / self.std
        P = np.c_[np.ones([n, 1]), P]

        reg_matrix = self.reg_lambda * np.eye(self.degree + 1)
        reg_matrix[0, 0] = 0 # avoid regularization on offset term
        try: 
            self.weight = np.linalg.solve(P.T @ P + reg_matrix, P.T @ y)
        except np.linalg.LinAlgError as e:
            # Possible to get singluar matrix when reg_lambda=0 as when
            # running generateLearningCurve(X, y, 1, 0)
            self.weight = np.linalg.lstsq(P.T @ P + reg_matrix, P.T @ y, rcond=None)[0]


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)
        P = self.polyfeatures(X, self.degree)
        P = (P - self.mean) / self.std
        P = np.c_[np.ones([n, 1]), P]

        # predict
        return P @ self.weight


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    if a.shape != b.shape or (a.ndim > 1 and a.shape[1] != 1) or (b.ndim > 1 and b.shape[1] != 1):
        raise ValueError(f"a and b must have the same shape: (n, 1).\n" +
                        f"a.shape: {a.shape}, b.shape: {b.shape}")
    err = a - b
    f = err.T @ err / len(err)
    return float(f)


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained 
            by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by 
            Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used 
        for training. THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the 
        learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    for i in range(n):
        xt = Xtrain[0:(i+1)]
        yt = Ytrain[0:(i+1)]
        try:
            model.fit(xt, yt)
        except:
            print(f"i={i}\nXtrain[0:(i+1)]: {xt}\nYtrain[0:(i+1)]: {yt}")
        errorTrain[i] = mean_squared_error(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = mean_squared_error(model.predict(Xtest), Ytest)
    return errorTrain, errorTest

if __name__ == "__main__":
    degree = 4
    reg_lambda = 0
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = np.ones_like(X) * 2
    expected = np.array([2, 0, 0, 0, 0])

    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    model.fit(X, y)
    actual = (
        model.weight.squeeze()
    )
    print(f"actual: {actual}")
