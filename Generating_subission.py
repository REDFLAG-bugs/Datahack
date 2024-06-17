from data_exploration import submission_format
import joblib

# Load the test features
test_features_preprocessed = joblib.load('test_features_preprocessed.pkl')

# Load the best model (e.g., Random Forest)
model = joblib.load('rf_model.pkl')

# Make predictions on the test data
test_preds = model.predict(test_features_preprocessed)

# Prepare the submission file
submission = submission_format.copy()
submission['xyz_vaccine'] = test_preds  # Adjust the column name as needed

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
