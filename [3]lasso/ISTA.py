from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with 
    calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when 
            to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, 
        second represents bias.
    
    """            
    v = (X @ weight - y + bias)    
    b = bias - 2*eta*v.sum()
    w = weight - 2*eta*(X.T @ v)    
    #print("####W####:", w, "####2*eta*_lambda####:", 2*eta*_lambda)

    # soft thresholding
    w[np.abs(w) <= 2*eta*_lambda] = 0 # must come first!
    w[w < -2*eta*_lambda] += 2*eta*_lambda
    w[w > 2*eta*_lambda] -= 2*eta*_lambda    
    return w, b

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """            
    n, d = X.shape        

    v = (X @ weight - y + bias)    
    return (v @ v) + _lambda * np.linalg.norm(weight, ord=1)


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing 
            predicted weights, and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None
    w = start_weight
    b = start_bias
    while old_w is None or not convergence_criterion(w, old_w, b, old_b, convergence_delta):
        old_w = np.copy(w)
        old_b = np.copy(b)
        w, b = step(X, y, w, b, _lambda, eta)
        max_delta = np.abs(np.hstack((w - old_w, b - old_b))).max()
    return w, b


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it 
    to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """    
    return np.abs(weight - old_w).max() <= convergence_delta


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    import numpy as np

    n = 500 # number of data points
    d = 1000 # dimension of the input features
    k = 100 # number of non-zero true weights/number of relevant features
    w_true = np.hstack((np.linspace(1/k, 1, num=k), np.zeros(d-k)))    

    rng = np.random.default_rng()
    X = rng.normal(size=(n, d)) # no need for standardization
    e = rng.normal(size=n)
    y = X @ w_true + e

    _lambda = 2*np.abs(X.T @ (y - y.mean())).max()
    num_non_zero = 0
    lambdas, num_non_zeros = [], []
    fdr, tpr = [], []
    while num_non_zero < d:
        w, b = train(X, y, _lambda, eta=2e-5, convergence_delta=1e-4)                
        num_non_zero = np.count_nonzero(w)
        lambdas.append(_lambda)
        num_non_zeros.append(num_non_zero)

        np.count_nonzero(w[0:k])
        if num_non_zero == 0:
            fdr.append(0)
            tpr.append(0)
        else:
            fdr.append(np.count_nonzero(w[k:d]) / num_non_zero)
            tpr.append(np.count_nonzero(w[0:k]) / k)
        _lambda *= 0.5

    plt.tight_layout()
    plt.plot(lambdas, num_non_zeros)
    plt.xscale("log")
    plt.xlabel("$\lambda$")
    plt.ylabel("Number of non-zero weights")
    plt.title("Lasso Regularization")
    plt.show()

    plt.plot(fdr, tpr)
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.title("True Positive Rate vs False Discovery Rate")
    plt.show()

def main_test():
    """
    Use all of the functions above to make plots.
    """
    import numpy as np

    n = 500 # number of data points
    d = 1000 # dimension of the input features
    k = 100 # number of non-zero true weights/number of relevant features
    sigma = 1
    _lambda = 200

    w_true = np.hstack((np.linspace(1/k, 1, num=k), np.zeros(d-k)))    

    rng = np.random.default_rng(seed=42)
    
    

    # xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)

    # fig, axs = plt.subplots(2, 3)
    # dis = (0, 1, 2, 3, 4, 5)
    # for ax, di in zip(axs.flat, dis):
    #     ax.plot(X[:,di], y, 'or', label=f"raw data[{di}]")
    #     ax.plot(X[:,di], y_opt, label=f"de-noised data[{di}]")
    #     ax.plot(X[:,di], X[:,di]*w_true[di], label=f"contribution to y")
    #     ax.legend()
    #     ax.set_title(f"w_true[{di}] = {w_true[di]:.2f}")
    #     ax.set(xlabel='X', ylabel='Y')
    # plt.show()

    # plt.plot(X[:,0], y, 'or', label="raw data[0]")
    # plt.plot(X[:,0], y_opt, label="de-noised data[0]")
    # plt.legend()
    # plt.show()
    # plt.plot(X[:,1], y, label="raw data[1]")
    # plt.plot(X[:,1], y_opt, label="de-noised data[1]")
    # plt.show()

    # X = rng.uniform(0, 10, size=(n, d))
    # mean = np.mean(X, axis=0, keepdims=True)
    # std = np.std(X, axis=0, keepdims=True)
    # std[std == 0] = 1 # avoid division by zero, just in case
    # X = (X - mean) / std

    X = rng.normal(size=(n, d))
    e = rng.normal(scale=sigma, size=n)
    y = X @ w_true + e
    y_opt = X @ w_true
    w, b = train(X, y, _lambda, eta=0.0001, convergence_delta=0.01)
    print("lol")        
    y_pred = X @ w + b    

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(y_opt,  y_pred, '.b')
    ax1.plot((y_pred.min(), y_pred.max()), (y_pred.min(), y_pred.max()), '--r')
    ax1.set_title("predictions vs de-noised data")
    ax2.plot(y, y_pred, '.b')
    ax2.plot((y_pred.min(), y_pred.max()), (y_pred.min(), y_pred.max()), '--r')
    ax2.set_title("predictions vs raw data")
    ax3.plot(w_true, w, '.b', alpha=0.2)
    ax3.plot((w.min(), w.max()), (w.min(), w.max()), '--r')
    ax3.set_title("predicted w vs true w")
    print(f"minimum loss={loss(X, y, w_true, 0.0, _lambda):.2f}, model loss={loss(X, y, w, b, _lambda):.2f}")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
