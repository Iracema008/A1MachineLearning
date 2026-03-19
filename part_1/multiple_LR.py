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

print("train rsme: ", train_rmse)
print("test rsme: ", test_rmse)

'''Normalize variables '''
columns = ['T', 'P', 'TC', 'SV']
mean = df[columns].mean()
standard_dev = df[columns].std()

normalized_df = df.copy()
normalized_df[columns] = (df[columns] - mean) / standard_dev

# Treat any value greater than 2 standard deviations from the mean as an outlier
out = (normalized_df[columns].abs() <= 2).all(axis=1)
remaining = normalized_df[out]

'''Retrain the linear regression model using the normalized dataset by computing 𝒘𝒘� as you did in step'''
X_norm = remaining[['T', 'P', 'TC', 'SV']].values
y_norm = remaining['Idx'].values

x_training_norm, x_testing_norm, y_training_norm, y_testing_norm = train_test_split(X_norm,y_norm, test_size=0.2, random_state=2)
'''Calculate training and testing RMSE.'''
norm_biased_x_training = bias_column(x_training_norm)
norm_biased_x_testing = bias_column(x_testing_norm)

w_hat_norm = least_squared(norm_biased_x_training, y_training_norm)
train_rmse_norm = rmse(norm_biased_x_training, y_training_norm, w_hat_norm)
test_rmse_norm  = rmse(norm_biased_x_testing,  y_testing_norm,  w_hat_norm)

print("normalized training rmse: ", train_rmse_norm)
print("normalized testing rmse: ", test_rmse_norm)