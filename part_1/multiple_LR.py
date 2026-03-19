import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# T = Tempreture, input x1
# P = Pressure, input x2
# TC = Criticial Tempreture?, x3
# SV = Specific Volume, x4
# Idx = Index, output

df = pd.read_csv('GasProperties.csv')
x = df[['T', 'P', 'TC', 'SV']].values
y =df['Idx'].values

x_training, x_testing, y_training, y_testing = train_test_split(x,y, test_size=0.2, random_state=2)

''' Implement the least square method '''
def bias_column(X):
    col_ones = np.ones((X.shape[0],1))
    return np.hstack([col_ones, X])

def least_squared(X,y):
    # our function to calculate weights, the inverse helps scale weights
    return np.linalg.inv(X.T @ X) @ X.T @ y

biased_x_training = bias_column(x_training)
biased_x_testing = bias_column(x_testing)

# w hat -> set of weights
w_hat = least_squared(biased_x_training, y_training)

'''Compute training and testing RMSE.'''
def rmse(X, y, w):
    # rmse is sqrt of mean( (actual-predicted)^2)
    predicted_vals = X @ w
    difference = (y - predicted_vals) ** 2
    mean = np.mean(difference)
    return np.sqrt(mean)

train_rmse = rmse(biased_x_training, y_training, w_hat)
test_rmse = rmse(biased_x_testing, y_testing, w_hat)

print("testing rsme: ", train_rmse)
print("training rsme: ", test_rmse)

'''Normalize variables '''
