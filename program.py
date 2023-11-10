# IMPORT LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND PROCESSING

# Loading the heart data from a CSV file to a Pandas dataframe
# This dataset contains information related to heart disease
heart_data = pd.read_csv('C:\\Users\\mistr\\OneDrive\\Documents\\Rohan\\Side\\Heart Disease Detection\\data.csv')

# Displaying the first and last five rows to understand the structure of the data
print(heart_data.head())
print(heart_data.tail())

# Exploring the dataset to understand its characteristics
# Checking for the number of rows and columns, information, missing values, and statistical measures
# Investigating the distribution of the target variable 'target'
print(heart_data.shape)
print(heart_data.info())
print(heart_data.isnull().sum())
print(heart_data.describe())
print(heart_data['target'].value_counts())

# SPLITTING THE FEATURES AND TARGET

# Splitting the dataset into features (x) and the target variable (y)
# Features contain information about the individuals, and the target is the presence or absence of heart disease
x = heart_data.drop(columns = 'target', axis = 1)
y = heart_data['target']
print(x)

# SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

# Splitting the data into training and testing sets for model evaluation
# Using 80% for training and 20% for testing, ensuring a stratified split for balanced target distribution
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)
print(x.shape, x_train.shape, x_test.shape)

# MODEL TRAINING

# Training a Logistic Regression model on the training data
# Logistic Regression is chosen for binary classification tasks like predicting heart disease
model = LogisticRegression()
model.fit(x_train, y_train)

# MODEL EVALUATION

# Evaluating the model performance using accuracy as the metric
# Accuracy represents the proportion of correctly predicted instances in the dataset

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on Training Data: ', training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on Test Data: ', test_data_accuracy)

# BUILDING A PREDICTIVE SYSTEM

# Building a predictive system to make predictions on new data
# Using an example input and interpreting the prediction
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have a Heart Disease!")
else:
    print("The Person has a Heart Disease!")
