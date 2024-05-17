import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

# Generate synthetic fMRI data with 6786 ROIs and 100 time points
num_rois = 6786
num_time_points = 100

# Create random synthetic fMRI data
np.random.seed(42)
data = np.random.rand(num_time_points, num_rois)

# Compute connectivity matrix
conn_est = ConnectivityMeasure(kind='correlation')
conn_matrix = conn_est.fit_transform([data])[0]

# Save the connectivity matrix as a CSV file
csv_filename = 'connectivity_matrix.csv'
pd.DataFrame(conn_matrix).to_csv(csv_filename, index=False)

print(f"Connectivity matrix saved as {csv_filename}")
