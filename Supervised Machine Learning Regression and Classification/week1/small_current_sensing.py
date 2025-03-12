"""
implement gradient decent for linear regression
"""
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.ion()              # enable interactive mode

from lab_utils_uni import plt_contour_wgrad, plt_divergence, plt_gradients, plt_gradients_new
#from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

def compute_cost_vectorized(x, y, w, b):
    f_wb = w * x + b
    cost = (f_wb - y) ** 2
    total_cost = (1 / (2 * x.shape[0])) * np.sum(cost)

    return total_cost

def compute_gradient_vectorized(x, y, w, b):
    f_wb = w * x + b
    dj_dw_i = (f_wb - y) * x
    dj_db_i = (f_wb - y)

    dj_dw = np.sum(dj_dw_i)/dj_dw_i.shape[0]  #np.mean(dj_dw_i)
    dj_db = np.sum(dj_db_i)/dj_db_i.shape[0]  #np.mean(dj_db_i)

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 

    w = copy.deepcopy(w_in) # avoid modifying global w_in
    b = b_in

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []

    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        if np.any(np.isnan([w, b])) or np.any(np.isinf([w, b])):
            print(f"Numerical issue at iteration {i}: w={w}, b={b}")
            break

        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(x, y, w , b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history #return w and J,w history for graphing


def predict_original_scale(x_new, w, b, mu_x, sigma_x, mu_y, sigma_y):
    # Normalize the new x
    x_new_norm = (x_new - mu_x) / sigma_x

    # Predict in normalized space
    y_pred_norm = w * x_new_norm + b

    # Denormalize the prediction
    y_pred = y_pred_norm * sigma_y + mu_y

    return y_pred

def denormalize_parameters(w_norm, b_norm, mu_x, sigma_x, mu_y, sigma_y):
    """
    Convert normalized weights and bias to real-scale values.

    Parameters:
    - w_norm: Weight trained in normalized space
    - b_norm: Bias trained in normalized space
    - mu_x: Mean of x_train
    - sigma_x: Standard deviation of x_train
    - mu_y: Mean of y_train
    - sigma_y: Standard deviation of y_train

    Returns:
    - w_real: Weight in original scale
    - b_real: Bias in original scale
    """
    w_real = (w_norm * sigma_y) / sigma_x
    b_real = mu_y - (w_norm * sigma_y * mu_x / sigma_x) + (b_norm * sigma_y)
    
    return w_real, b_real



x_train = np.array([3, 14, 25, 36, 47, 58, 112])
y_train = np.array([0, 10, 20, 30, 40, 50, 100])

plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Small Output Current Sensing")
plt.ylabel('Applied Current (mA)')
plt.xlabel('Measured Current (mA)')
plt.legend()
plt.show()


##############################################################
# Scale down to avoid overflow (remember to scale back the prediction later)
mu_x = np.mean(x_train)
sigma_x = np.std(x_train)

mu_y = np.mean(y_train)
sigma_y = np.std(y_train)

x_train_norm = (x_train - mu_x) / sigma_x
y_train_norm = (y_train - mu_y) / sigma_y

plt_gradients_new(x_train_norm, y_train_norm, compute_cost_vectorized, compute_gradient_vectorized)
plt.show()
##############################################################

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-3
# run gradient descent
w_final_norm, b_final_norm, J_hist, p_hist = gradient_descent(x_train_norm ,y_train_norm, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost_vectorized, compute_gradient_vectorized)
print(f"(w,b) found by gradient descent: ({w_final_norm:8.4f},{b_final_norm:8.4f})")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()


# Get real model parameters
w_final, b_final = denormalize_parameters(w_final_norm, b_final_norm, mu_x, sigma_x, mu_y, sigma_y)


# Predictions
sample_set = 6
x_new = x_train[sample_set]
#y_pred = predict_original_scale(x_new, w_final_norm, b_final_norm, mu_x, sigma_x, mu_y, sigma_y)
y_pred = w_final * x_new + b_final
print(f"Predicted Input Current for Measured={x_new}: {y_pred:.2f} (mA) : Error(%)= {abs((y_pred-y_train[sample_set])*100/y_train[6]):.2f} ")

sample_set = 1
x_new = x_train[sample_set]
#y_pred = predict_original_scale(x_new, w_final, b_final, mu_x, sigma_x, mu_y, sigma_y)
y_pred = w_final * x_new + b_final
print(f"Predicted Input Current for Measured={x_new}: {y_pred:.2f} (mA) : Error(%)= {abs((y_pred-y_train[sample_set])*100/y_train[6]):.2f} ")


for i in range(x_train.shape[0]):
    sample_set = i
    x_new = x_train[sample_set]
    y_pred = w_final * x_new + b_final
    print(f"Predicted Input Current for Measured={x_new}: {y_pred:.2f} (mA) : Error(%)= {abs((y_pred-y_train[sample_set])*100/y_train[6]):.2f} ")


# Plotting
# - show the progress of gradient descent during its execution by plotting the 
#   cost over iterations on a contour plot of the cost(w,b).
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train_norm, y_train_norm, p_hist, ax)


fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train_norm, y_train_norm, p_hist, ax, w_range=[0.9, 1.15, 0.001], b_range=[-0.05, 0.01, 0.001],
            contours=[1,5,10,20],resolution=0.5)


plt.show(block=True)    # prevents script from exiting immediately