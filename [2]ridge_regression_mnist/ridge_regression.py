import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

from utils import load_dataset, problem


@problem.tag("hw1-A")
def train(X: np.ndarray, Y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`X`), targets (`Y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        X (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        Y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `X` and `Y` with hyperparameter `_lambda`.
    """
    n, d = X.shape
    n_y, k = Y.shape
    if n != n_y:
        raise ValueError(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

    ret = np.linalg.solve(X.T @ X + _lambda * np.eye(d), X.T @ Y)
    if ret.shape != (d, k):
        raise ValueError(f"ret.shape: {ret.shape}, (d, k): {(d, k)}")
    return ret


@problem.tag("hw1-A")
def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`X`), and weight matrix (`W`) to generate predicated class for each 
    observation in X.

    Args:
        X (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        W (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    n, d = X.shape
    d_w, k = W.shape
    if d != d_w:
        raise ValueError(f"X.shape: {X.shape}, W.shape: {W.shape}")
    return np.argmax(X @ W, axis=1)


@problem.tag("hw1-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th
     element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    n = len(y)
    ret = np.zeros((n, num_classes))
    ret[np.arange(n), y] = 1
    return ret


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print(y_test_pred[0:10])

    fig, axs = plt.subplots(2, 5)
    i = 0
    for ax in axs.flat:
        while y_test_pred[i] == y_test[i]:
            i += 1
        ax.imshow(x_test[i].reshape(28, 28), cmap="Purples")
        ax.set_title(f"Label: {y_test[i]}, Prediction: {y_test_pred[i]}")
        i += 1
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    
    plt.tight_layout()
    plt.show()

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")


if __name__ == "__main__":
    main()
