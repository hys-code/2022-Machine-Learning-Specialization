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

def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """
    m, n = X.shape
    
    ### START CODE HERE ###
    z = X @ w + b
    cost = -y*np.log(sigmoid(-z)) - (1-y)*np.log(1-sigmoid(-z))
    
    cost_lambda = lambda_*w**2/(2*m) 
    
    total_cost = np.sum(cost)/m + np.sum(cost_lambda)
    ### END CODE HERE ### 

    return total_cost

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


# Test for compute_cost()
m, n = X_train.shape


# Verify compute_cost()
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


plt.show(block=True)

