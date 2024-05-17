# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nilearn
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.datasets import fetch_abide_pcp
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Fetching fMRI data from the NYU repository with respect to the AAL atlas
data = fetch_abide_pcp(derivatives=['rois_aal'], SITE_ID=['NYU'])

# Pre-processing the data
conn_est = ConnectivityMeasure(kind='correlation')
conn_matrices = conn_est.fit_transform(data['rois_aal'])
X = [sym_matrix_to_vec(conn_matrix) for conn_matrix in conn_matrices]
X = np.vstack(X)

# Extracting diagnostic labels (DX_GROUP) from the dataset
y = data.phenotypic['DX_GROUP']

# Standardizing the features using StandardScaler
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# Converting diagnostic labels: 2 to -1 (binary classification)
y[y == 2] = -1

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Training the Random Forest model on the training set
rf_model.fit(X_train, y_train)

# Saving the Random Forest model using joblib
model_filename = 'rf_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"Random Forest model saved to {model_filename}")

# Evaluating the accuracy on the test set
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy}")
