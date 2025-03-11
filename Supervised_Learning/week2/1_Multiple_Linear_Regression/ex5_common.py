import numpy as np
import copy
import math

def compute_cost_vectorized(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)

def compute_gradient_vectorized(X, y, w, b): 
    m, n = X.shape
    predictions = X @ w + b
    errors = predictions - y
    dj_dw = (X.T @ errors) / m
    dj_db = np.sum(errors) / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
    return w, b, J_history

# Test case: scalar X, w
X = np.array([[2]])    # shape (1, 1)
y = np.array([4])      # shape (1,)
w = np.array([0.5])    # shape (1,)
b = 0.1

w_final, b_final, J_hist = gradient_descent(X, y, ww, b,
                                            compute_cost_vectorized,
                                            compute_gradient_vectorized,
                                            alpha=0.1,
                                            num_iters=10)
print("Final w:", w_final)
print("Final b:", b_final)
