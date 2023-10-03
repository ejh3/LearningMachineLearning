if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    def mse(a, b):
        return ((a-b)**2).mean().squeeze()
    import pandas as pd
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")
    
    f_names = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    f_indices = [df_train.columns.get_loc(f) for f in f_names]
    eta = 2e-5 # learning rate

    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values
    
    lambdas = []
    num_non_zeros = []
    f_coeffs = np.array([[]]).reshape(0,len(f_names))
    mses = np.array([[]]).reshape(0,2) # train, test MSEs
    w_old, b_old = None, None

    _lambda = 2*np.abs(X.T @ (y - y.mean())).max()
    while _lambda >= 0.01:
        w, b = train(X, y, _lambda, eta, 1e-4, w_old, b_old)
        num_non_zero = np.count_nonzero(w)
        print(f"num_non_zero={num_non_zero}, lambda={_lambda:.3e}")

        lambdas.append(_lambda)
        num_non_zeros.append(num_non_zero)
        f_coeffs = np.vstack((f_coeffs, w[f_indices]))
        mses = np.vstack((mses, np.array([mse(X @ w + b, y), mse(X_test @ w + b, y_test)])))
        
        w_old, b_old = w, b
        _lambda *= 0.5
    
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(lambdas, num_non_zeros)
    ax[0].set_ylabel("number of non-zero weights")
    ax[0].set_title("Lasso Regularization Overview")

    ax[1].plot(lambdas, f_coeffs)
    ax[1].set_title("Lasso Regularization Paths")
    ax[1].legend(f_names)

    ax[2].plot(lambdas, mses)
    ax[2].set_title("Mean Squared Errors")
    ax[2].legend(["train MSE", "test MSE"])
    plt.xlabel("$\lambda$")
    plt.xscale("log")
    plt.show()

    _lambda = 30
    w, b = train(X, y, _lambda, eta, 1e-4)
    labeled_weights = list(zip(w, df_train.columns[1:]))
    labeled_weights.sort(key=lambda x: x[0])
    print(f"Weights for λ={_lambda}:")
    for weight, label in labeled_weights:
        print(f"{label+':': <22} {weight: .3e}")

if __name__ == "__main__":
    main()
