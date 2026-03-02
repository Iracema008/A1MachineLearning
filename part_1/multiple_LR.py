import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('GasProperties.csv')
x = df.drop('Score', axis=1)
y =df['Score']
x_training, x_testing, y_training, y_testing = train_test_split(x,y, test_size=0.2, random_state=2)
