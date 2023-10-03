import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    fig, axs = plt.subplots(2, 3)

    # regression with degree = d
    d = 8
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    reg_lambdas = [0, 1e-7, 1e-5, 1e-3, 1, 100]
    for ax, reg_lambda in zip(axs.flat, reg_lambdas):
        model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
        model.fit(X, y)

        ypoints = model.predict(xpoints)
        ax.plot(xpoints, ypoints, label=r"$\lambda = $ " + str(reg_lambda))
        ax.plot(X, y, "rx", label="Training data")
        ax.set_title(f"PolyRegression with λ = {reg_lambda}")
        ax.set(xlabel='X', ylabel='Y')
        ax.label_outer()

    plt.show()
