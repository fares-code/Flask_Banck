# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector
import joblib

# Load data
filtered_df = pd.read_csv("BankMarketingAfterClean.csv")

# Split data
X = filtered_df.drop('deposit', axis=1)
y = filtered_df['deposit']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Get numerical columns
numerical_cols = x_train.select_dtypes(include='number').columns

# Standardize data
scaler = StandardScaler()
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()
x_train_scaled[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
x_test_scaled[numerical_cols] = scaler.transform(x_test[numerical_cols])

# Train Gaussian Naive Bayes with backward feature selection
backward_gaussian_nb = GaussianNB()
backward_gaussian_nb_sfs = SequentialFeatureSelector(
    backward_gaussian_nb, 
    n_features_to_select="auto", 
    direction='backward', 
    scoring='accuracy'
)
backward_gaussian_nb_sfs.fit(x_train_scaled[numerical_cols], y_train)

# Get selected features
selected_features = x_train_scaled[numerical_cols].columns[backward_gaussian_nb_sfs.get_support()]
print("Selected Features:", selected_features.tolist())

# Train final model on selected features
x_train_selected = x_train_scaled[selected_features]
backward_gaussian_nb.fit(x_train_selected, y_train)

# Save model
joblib.dump(backward_gaussian_nb, 'backward_gaussian_nb_model.joblib')
print("Model saved successfully!")

# Optional: Save scaler too
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved successfully!")
