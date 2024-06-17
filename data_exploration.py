import pandas as pd

# Load the datasets
train_features = pd.read_csv('dataset and all/training_set_features.csv')
train_labels = pd.read_csv('dataset and all/training_set_labels.csv')
test_features = pd.read_csv('dataset and all/test_set_features.csv')
submission_format = pd.read_csv('dataset and all/submission_format.csv')

# Display the first few rows of each dataset
print("Training Features:")
print(train_features.head())
print("\nTraining Labels:")
print(train_labels.head())
print("\nTest Features:")
print(test_features.head())
print("\nSubmission Format:")
print(submission_format.head())

# Check for missing values
print("\nMissing Values in Training Features:")
print(train_features.isnull().sum())
print("\nMissing Values in Training Labels:")
print(train_labels.isnull().sum())
print("\nMissing Values in Test Features:")
print(test_features.isnull().sum())

# Check the data types
print("\nData Types in Training Features:")
print(train_features.dtypes)
print("\nData Types in Training Labels:")
print(train_labels.dtypes)
print("\nData Types in Test Features:")
print(test_features.dtypes)
