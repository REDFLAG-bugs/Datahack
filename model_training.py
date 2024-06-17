import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
from data_exploration import train_labels
from data_preprocessing import numerical_cols, categorical_cols, preprocessor
# Load preprocessed data
train_features_preprocessed = joblib.load('train_features_preprocessed.pkl')
test_features_preprocessed = joblib.load('test_features_preprocessed.pkl')

# Extract labels
y = train_labels['xyz_vaccine']  # Use the correct label column

# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=10000, random_state=42)
logreg.fit(train_features_preprocessed, y)

# Evaluate Logistic Regression model
logreg_preds = logreg.predict(train_features_preprocessed)
logreg_acc = accuracy_score(y, logreg_preds)
logreg_f1 = f1_score(y, logreg_preds)
logreg_roc_auc = roc_auc_score(y, logreg_preds)

print(f"Logistic Regression Accuracy: {logreg_acc}")
print(f"Logistic Regression F1 Score: {logreg_f1}")
print(f"Logistic Regression ROC AUC: {logreg_roc_auc}")

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features_preprocessed, y)

# Evaluate Random Forest model
rf_preds = rf.predict(train_features_preprocessed)
rf_acc = accuracy_score(y, rf_preds)
rf_f1 = f1_score(y, rf_preds)
rf_roc_auc = roc_auc_score(y, rf_preds)

print(f"Random Forest Accuracy: {rf_acc}")
print(f"Random Forest F1 Score: {rf_f1}")
print(f"Random Forest ROC AUC: {rf_roc_auc}")

# Save the models
joblib.dump(logreg, 'logreg_model.pkl')
joblib.dump(rf, 'rf_model.pkl')

# Feature importances from Random Forest
importances = rf.feature_importances_
feature_names = np.array(numerical_cols.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)))
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f in range(len(importances)):
    print(f"{f+1}. feature {indices[f]} ({importances[indices[f]]})")
