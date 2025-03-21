"""
Utilize scikit-learn to implement linear regression using a close solution based on the normal equation
"""
import numpy as np
np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([3, 25, 47, 69, 91, 112])    # measured current
y_train = np.array([0, 20, 40, 60, 80, 100])    # true current (prediction)

x_verify = np.array([14, 36, 58, 80, 102])
y_verify = np.array([10, 30, 50, 70, 90])

linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)

b = linear_model.intercept_
w = linear_model.coef_

print(f"w = {w:}, b = {b:0.2f}")

#####################################################################
# verification

print(" ********* Verify Values ********* ")
print("Input Values: " + str(x_verify.tolist()))
print("True Values:  " + str(y_verify.tolist()))

x_verify_reshaped = x_verify.reshape(-1, 1)     # Scikit-learn expects 2D array as input

predictions = linear_model.predict(x_verify_reshaped)
formatted_predictions = [f"{pred:.2f}" for pred in predictions]
print("Prediction Values: ", formatted_predictions)

    
input("Press Enter to exit...")

