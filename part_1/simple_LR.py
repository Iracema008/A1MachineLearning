
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

'''
Supervised learning for linear regression [40 points]
(1) Use the training dataset “study_data.csv” that consists of hours (number of hours studied) and score
(exam score) to find a functional relationship between the number of study hours and exam score.

Tasks to write a program “simple_LR.py”
(a) Split the dataset into 80% training set and 20% testing set.
(b) Fit a linear regression model using “LinearRegression()” from sklearn.linear_model package.
(c) Computes training and testing errors using the Rooted Mean Squared Error 𝑹𝑹𝑹𝑹𝑹𝑹𝑹𝑹 = � ∑ (𝑦𝑦 𝑖𝑖 −𝑦𝑦𝚤𝚤� )2
𝑖𝑖
𝑁𝑁
where 𝑦𝑦𝑖𝑖 is actual 𝑦𝑦 value in the training dataset, 𝑦𝑦𝚤𝚤� is the predicted 𝑦𝑦 value for each 𝑥𝑥 value, and N
is the total number of examples in the dataset.
(d) Write the learned function in polynomial form: 𝑦𝑦 = 𝑤𝑤0 + 𝑤𝑤1 𝑥𝑥.

'''

df = pd.read_csv('study_data.csv')
x = df.drop('Score', axis=1)
y =df['Score']
x_training, x_testing, y_training, y_testing = train_test_split(x,y, test_size=0.2, random_state=12)

linear_reg = LinearRegression()
linear_reg.fit(x_training, y_training)
predicted_values = linear_reg.predict(x_testing)

print("\nThe Rooted Mean Squared Error is ", np.sqrt(np.mean((y_testing - predicted_values) ** 2)))
print("The Learned Function is y = ", linear_reg.intercept_, " +", linear_reg.coef_[0], "x\n")
#rmse = root_mean_squared_error(y, )