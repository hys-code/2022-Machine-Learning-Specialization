import numpy as np
import matplotlib.pyplot as plt
plt.ion()   # enable interactive mode

from utils import *
import copy
import math

# z can be a scalar or array of numbers
def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, lambda_= 0):
    m, n = X.shape
    z = X @ w + b
    s = sigmoid(z)
    
    cost = -y*np.log(s) - (1-y)*np.log(1-s)
    reg_cost = (lambda_/2*m) * np.sum(w**2)
    total_cost = (1/m)*np.sum(cost) + reg_cost
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_=0):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0

    z = X @ w + b
    s = sigmoid(z)
    err = s - y

    dj_db = (1/m)*np.sum(err) 
    #dj_dw = (1/m)*np.sum(np.dot(err, X)) + (lambda_/m)*w
    dj_dw = (1/m)*(X.T @ err) + (lambda_/m)*w

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      
    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing


X_train, y_train = load_data("data/ex2data1.txt")

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))


# Visualize data
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")

# test for scalar (expected value: 0.5)
print ("sigmoid(0) = " + str(sigmoid(0)))

# test for vector 
print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS 
from public_tests import *
sigmoid_test(sigmoid)


# ********** Test for compute_cost() **********
m, n = X_train.shape

initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))  # Expect: 0.0693

# Compute and display cost with non-zero w
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w,b: {:.3f}'.format(cost)) # expect: 0.219

# UNIT TESTS
compute_cost_test(compute_cost)


# ********** Test for compute_gradient() **********
# Compute and display gradient with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}' )           # Expect: -0.1
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )


# Compute and display cost and gradient with non-zero w
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test_w:', dj_db)
print('dj_dw at test_w:', dj_dw.tolist())

# UNIT TESTS    
compute_gradient_test(compute_gradient)


# ********** Test for gradient_descent() **********
np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

plot_decision_boundary(w, b, X_train, y_train)

plt.show(block=True)

