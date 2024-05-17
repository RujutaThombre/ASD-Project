import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import joblib

# Generate synthetic data with a 4x4 matrix and at least 10 samples
num_samples = 100
num_features = 4

# Create random synthetic connectivity matrices
np.random.seed(42)
synthetic_data = np.random.rand(num_samples, num_features, num_features)

# Flatten the matrices for compatibility with the Random Forest model
X = [matrix.flatten() for matrix in synthetic_data]
X = np.vstack(X)

# Generate random labels (assuming binary classification for demonstration)
y = np.random.choice([0, 1], size=num_samples)

# Creating a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Standardize the features using StandardScaler
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# Training the Random Forest model
rf_model.fit(X, y)

# Saving the Random Forest model using joblib
model_filename = 'rf_model_4x4.joblib'
joblib.dump(rf_model, model_filename)
print(f"Random Forest model saved to {model_filename}")
