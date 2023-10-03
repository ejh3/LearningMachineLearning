import csv
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

def calc_error(
    R_hat: np.ndarray, true_ratings: np.ndarray
) -> np.ndarray:
    """
    Compute the Mean squared error of the estimated R_hat on the test set R_test.

    Args:
        R_hat (np.ndarray): Array of shape (m, n) where entry R_hat[i, j] is the estimated rating 
            of user j for movie i.
        true_ratings (np.ndarray): Array of shape (k,) with each row of the form (j, i, s) where j 
            is the user index, i is the movie index, and s is the user's true rating.

    Returns:
        float: Mean squared error of the estimations in R_hat versus the true ratings in R_test.
    """
    return np.mean((R_hat[true_ratings[:, 1], true_ratings[:, 0]] - true_ratings[:, 2]) ** 2)


# load data
data = []
with open('./u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)  # num_observations = 100,000
num_users = max(data[:,0])+1  # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1  # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

print(f"Successfully loaded 100K MovieLens dataset with",
      f"{len(train)} training samples and {len(test)} test samples")


print("\n-------------------- A4 Part a --------------------")
count_vec = np.bincount(train[:, 1], minlength=num_items)
sum_vec = np.bincount(train[:, 1], weights=train[:, 2], minlength=num_items)
count_vec[count_vec == 0] = 1

R_hat_avg = np.outer((sum_vec / count_vec), np.ones(num_users))
print(f"Viewer Average Rating Estimator Test Error: {calc_error(R_hat_avg, test)}")


print("\n-------------------- A4 Part b --------------------")
R_twiddle = np.zeros((num_items, num_users))
R_twiddle[train[:, 1], train[:, 0]] = train[:, 2]

d_vals = [1, 2, 5, 10, 20, 50]
train_errors = []
test_errors = []
svd_results = {}

for d in d_vals:
    print(f"Training d={d} with SVD...")
    U, S, VT = svds(R_twiddle, k=d)
    svd_results[d] = (U, np.diag(S), VT)
    R_hat_d = U @ np.diag(S) @ VT
    train_errors.append(calc_error(R_hat_d, train))
    test_errors.append(calc_error(R_hat_d, test))

plt.plot(d_vals, train_errors, label="Train")
plt.plot(d_vals, test_errors, label="Test")
plt.xlabel("d")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title("MSE vs d for SVD Approximation")
plt.xticks(d_vals)
plt.show()


print("\n-------------------- A4 Part c --------------------")
train_errors = []
test_errors = []
part_d_model = None
j_to_is = defaultdict(list)
i_to_js = defaultdict(list)
for j, i, s in train:
    j_to_is[j].append(i)
    i_to_js[i].append(j)

_lambda = 0.1
max_iter = 10
for d in d_vals:
    # Initialize U, V
    U = svd_results[d][0] @ sqrtm(svd_results[d][1])
    V = (sqrtm(svd_results[d][1]) @ svd_results[d][2]).T
    print(f"Training d={d} with alternating minimization...")
    for c in range(max_iter):
        # Fix V and solve for U
        for i in range(num_items):
            v_js = V[i_to_js[i]]
            U[i] = np.linalg.inv(v_js.T @ v_js + _lambda * np.eye(d)) \
                @ (v_js.T @ R_twiddle[i, i_to_js[i]])
        # Fix U and solve for V
        for j in range(num_users):
            u_is = U[j_to_is[j]]
            V[j] = np.linalg.inv(u_is.T @ u_is + _lambda * np.eye(d)) \
                @ (u_is.T @ R_twiddle[j_to_is[j], j])
        if c+1 in np.linspace(1, max_iter, num=4, endpoint=True, dtype=int):
            print(f"  ({c+1}/{max_iter}) train error={calc_error(U @ V.T, train)}")
    train_errors.append(calc_error(U @ V.T, train))
    test_errors.append(calc_error(U @ V.T, test))
    if test_errors[-1] <= 0.9:
        part_d_model = {"d": d, "U": U, "V": V, "Error": test_errors[-1]}
    print(f"test error={test_errors[-1]}\n")

plt.plot(d_vals, train_errors, label="Train")
plt.plot(d_vals, test_errors, label="Test")
plt.xlabel("d")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title(f"MSE vs d for Alternating Minimization, λ={_lambda}, {max_iter} iterations")
if part_d_model is not None:
    x, y = part_d_model['d'], part_d_model['Error']
    plt.annotate(f"(d={d_vals[1]}, Test Error={test_errors[1]:.3f})", 
        xy=(d_vals[1], test_errors[1]), xytext=(d_vals[1], test_errors[1] + 1), 
        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0), ha = 'left', va = 'bottom')
plt.xticks(d_vals)
plt.show()


print("\n-------------------- A4 Part d --------------------")
if part_d_model is None:
    print("Could not find a model with test error < 0.9")
else:
    print(f"Achieved a test error of {part_d_model['Error']:.3f} < 0.9 "
        + f"with d={part_d_model['d']}, λ={_lambda}, and {max_iter} iterations")
    print(f"U shape: {part_d_model['U'].shape},  V shape: {part_d_model['V'].shape}")
print()

# -------------------- A4 Part a --------------------
# Viewer Average Rating Estimator Test Error: 1.0635642005674517

# -------------------- A4 Part b --------------------
# Training d=1 with SVD...
# Training d=2 with SVD...
# Training d=5 with SVD...
# Training d=10 with SVD...
# Training d=20 with SVD...
# Training d=50 with SVD...

# -------------------- A4 Part c --------------------
# Training d=1 with alternating minimization...
#   (1/10) train error=0.9885860880210139
#   (4/10) train error=0.8331903876518322
#   (7/10) train error=0.8329817321221421
#   (10/10) train error=0.8329796980055187
# test error=0.9097656394173067

# Training d=2 with alternating minimization...
#   (1/10) train error=0.928758937078461
#   (4/10) train error=0.7462034132336507
#   (7/10) train error=0.7381360924415229
#   (10/10) train error=0.7366920205630639
# test error=0.8784575214522788

# Training d=5 with alternating minimization...
#   (1/10) train error=0.800737183772671
#   (4/10) train error=0.6068079617731252
#   (7/10) train error=0.5971425981977061
#   (10/10) train error=0.5933146623234159
# test error=1.005844399638465

# Training d=10 with alternating minimization...
#   (1/10) train error=0.6212598446274836
#   (4/10) train error=0.45359431852842375
#   (7/10) train error=0.4378583472900636
#   (10/10) train error=0.4311749027634305
# test error=1.3045473861235846

# Training d=20 with alternating minimization...
#   (1/10) train error=0.45581822977784614
#   (4/10) train error=0.2790402323713186
#   (7/10) train error=0.253881337267711
#   (10/10) train error=0.24259300125847214
# test error=1.8830965500911216

# Training d=50 with alternating minimization...
#   (1/10) train error=0.191591811366928
#   (4/10) train error=0.062353465614897116
#   (7/10) train error=0.04448590572590978
#   (10/10) train error=0.036616821819977614
# test error=2.888100300262298


# -------------------- A4 Part d --------------------
# Achieved a test error of 0.878 < 0.9 with d=2, λ=0.1, and 10 iterations
# U shape: (1682, 2),  V shape: (943, 2)