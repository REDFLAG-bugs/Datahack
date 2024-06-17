import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from data_exploration import train_features, test_features

# Identify numerical and categorical columns
numerical_cols = train_features.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_features.select_dtypes(include=['object']).columns

# Preprocessing for numerical data: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the training and test data
train_features_preprocessed = preprocessor.fit_transform(train_features)
test_features_preprocessed = preprocessor.transform(test_features)

# Save the preprocessed data and preprocessor
joblib.dump(train_features_preprocessed, 'train_features_preprocessed.pkl')
joblib.dump(test_features_preprocessed, 'test_features_preprocessed.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Shape of the preprocessed training features:", train_features_preprocessed.shape)
print("Shape of the preprocessed test features:", test_features_preprocessed.shape)
