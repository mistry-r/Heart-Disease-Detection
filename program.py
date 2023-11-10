#IMPORT LIBRARIES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#DATA COLLECTION AND PROCESSING

#loading the csv data to a Pandas dataframe
heart_data = pd.read_csv('C:\\Users\\mistr\\OneDrive\\Documents\\Rohan\\Side\\Heart Disease Detection\\data.csv')
#print the first five rows of the dataset
print(heart_data.head())
#print the last five rows of the dataset
print(heart_data.tail())
#number of rows and columns in the dataset
print(heart_data.shape)
#dataset information
print(heart_data.info())
#checking the dataset for missing values
print(heart_data.isnull().sum())
#statistical measures of the data
print(heart_data.describe())
#checking the distribution of the target variable
print(heart_data['target'].value_counts())

#SPLITTING THE FEATURES AND TARGET

x = heart_data.drop(columns = 'target', axis = 1)
y = heart_data['target']
print(x)

#SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)
print(x.shape, x_train.shape, x_test.shape)

#MODEL TRAINING

model = LogisticRegression()

model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy on Training Data: ', training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy on Test Data: ', test_data_accuracy)

input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

input_data_as_numpy_array = np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The Person does not have a Heart Disease!")
else:
    print("The Person has a Heart Disease!")
