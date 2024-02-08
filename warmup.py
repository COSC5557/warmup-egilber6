# looked at warmup-Brookluo code
# used chatgpt
# looked at these websites
# https://www.w3schools.com/python/python_ml_linear_regression.asp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# put data from data set into a pandas DataFrame
red = pd.read_csv('winequality-red.csv', sep=';', index_col=None)
white = pd.read_csv('winequality-white.csv', sep=';', index_col=None)

# concatenate the red wine and white wine into a single x and y
data_set = pd.concat([red, white], ignore_index=True)

# add the class to specify which wine is good or bad
data_set['class'] = data_set['quality'].apply(lambda x: 1 if x >= 6 else 0)

# separate the wines into the independent variables x and the depended variable and class y
X = data_set.iloc[:, :-2].values
Y = data_set.iloc[:, -2:].values

# seperate into training and testing sets, use the randome_state 
#to get reproductible results, take away or change to get different results
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# scale the data so the model can perform better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# choose model, in this case linear regression
model = LinearRegression()
# train the data
model.fit(X_train_scaled, Y_train[:,0])

# get the predictions of the model
Y_pred = model.predict(X_test_scaled)

# compare the predictions to the actual values
mse = mean_squared_error(Y_test[:,0], Y_pred)
r2 = r2_score(Y_test[:,0], Y_pred)

# get the class of the predicted values
Y_pred_class = np.where(Y_pred >= 6, 1, 0)

# the Mean Squared error is the sum of the squared errors divided by the
# number of data points used in the calculation
print(f'Mean Squared Error: {mse}')
# gives the R-squared value, which shows how well the independed variables in 
# the model correlate with the depended variable
print(f'R-squared: {r2}')

# see how accurate the model is to predicting the class of the wine
accuracy = accuracy_score(Y_test[:,1], Y_pred_class)
# provide a classification repo
classification_report_str = classification_report(Y_test[:,1], Y_pred_class)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_str)
#my ouput values
#Mean Squared Error: 0.5507405895964231
#R-squared: 0.2704702244055961
#Accuracy: 0.6661538461538462
#Classification Report:
#                precision    recall  f1-score   support
#
#           0       0.52      0.89      0.65       229
#           1       0.90      0.54      0.68       421
#
#    accuracy                           0.67       650
#   macro avg       0.71      0.72      0.67       650
#weighted avg       0.77      0.67      0.67       650
