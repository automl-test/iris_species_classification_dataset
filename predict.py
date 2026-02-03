"""
Prediction script for classification model

This script loads a trained model and makes predictions on new data.
"""

import pandas as pd
import pickle

# Load the trained model
with open('classification_model_20260204_021448.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    label_encoder = saved_data.get('label_encoder')

# Load new data for prediction
# TODO: Update this path to your new data
new_data = pd.read_csv('new_data.csv')

# Prepare data (same preprocessing as training)
X = pd.get_dummies(new_data, drop_first=True)
X = X.fillna(X.mean())

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict(X_scaled)

# Decode if classification with label encoder
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)

# Save predictions
result_df = pd.DataFrame({
    'predictions': predictions
})
result_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
print("\nFirst few predictions:")
print(result_df.head())
