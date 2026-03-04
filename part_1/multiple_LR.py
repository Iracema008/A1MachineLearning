import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# T = Tempreture, input x1
# P = Pressure, input x2
# TC = Criticial Tempreture?, y1
# SV = Specific Volume, y2

# Idx = Index, output

df = pd.read_csv('GasProperties.csv')
x = df[['T', 'P']]
y =df[['TC','SV', ]]

x_training, x_testing, y_training, y_testing = train_test_split(x,y, test_size=0.2, random_state=2)
correlation_vector = np.corrcoef(x_training, y_training)
w_hat = correlation_vector[0:2, 2:4] @ np.linalg.inv(correlation_vector[0:2, 0:2])

#np.linalg.lstsq(x_training, y_training)


