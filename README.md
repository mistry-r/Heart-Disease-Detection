# Heart Disease Prediction

This Python script focuses on predicting heart disease using logistic regression. It covers data collection, processing, model training, evaluation, and building a predictive system. Below is an overview of the code's functionality and components.

## Code Overview

### IMPORT LIBRARIES
Imports necessary libraries, including NumPy, Pandas, and scikit-learn's logistic regression model.

### DATA COLLECTION AND PROCESSING
Loads heart disease data from a CSV file into a Pandas dataframe and explores the dataset's characteristics. It includes checking for missing values, statistical measures, and investigating the distribution of the target variable.

### SPLITTING THE FEATURES AND TARGET
Splits the dataset into features (x) and the target variable (y), where features contain information about individuals, and the target represents the presence or absence of heart disease.

### SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA
Divides the data into training and testing sets for model evaluation. Uses an 80-20 split with stratification for a balanced target distribution.

### MODEL TRAINING
Trains a logistic regression model on the training data, chosen for binary classification tasks like predicting heart disease.

### MODEL EVALUATION
Evaluates the model's performance using accuracy as the metric, representing the proportion of correctly predicted instances in the dataset.

### BUILDING A PREDICTIVE SYSTEM
Builds a predictive system to make predictions on new data. Utilizes an example input and interprets the prediction.

## How to Use

To use this code:

1. Ensure you have the required Python libraries (NumPy, Pandas, scikit-learn) installed.
2. Place your heart disease data in a CSV file named 'data.csv'.
3. Execute the Python script. It will perform data processing, model training, and generate predictions.

Review the accuracy on both training and test data, and explore the predictions made by the built system.

Feel free to customize the script according to your dataset and specific requirements.

Enjoy predicting heart disease with your data!
