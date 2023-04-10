#Here's a new Python script for preprocessing the accelerometer data. This script includes applying a moving average filter to reduce noise, detecting and removing outliers, handling imbalanced data, and normalizing the data using sklearn's preprocessing tools.
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Moving average filter
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Remove outliers based on z-score
def remove_outliers(data, threshold=3):
    z_scores = np.abs(zscore(data))
    return data[(z_scores < threshold).all(axis=1)]

# Read segmented data from HDF5 file
with h5py.File('data.h5', 'r') as f:
    kieran_walking_segments = f['Kieran/walking'][:]
    kieran_jumping_segments = f['Kieran/jumping'][:]

# Apply moving average filter to reduce noise
window_size = 5
kieran_walking_filtered = moving_average(kieran_walking_segments, window_size)
kieran_jumping_filtered = moving_average(kieran_jumping_segments, window_size)

# Detect and remove outliers
kieran_walking_no_outliers = remove_outliers(kieran_walking_filtered)
kieran_jumping_no_outliers = remove_outliers(kieran_jumping_filtered)

# Handle imbalanced data (if necessary)
min_length = min(len(kieran_walking_no_outliers), len(kieran_jumping_no_outliers))
balanced_walking_data = kieran_walking_no_outliers[:min_length]
balanced_jumping_data = kieran_jumping_no_outliers[:min_length]

# Combine walking and jumping data
combined_data = np.concatenate((balanced_walking_data, balanced_jumping_data), axis=0)

# Normalize the data using StandardScaler from sklearn
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_data)

# Save the preprocessed data into a new HDF5 file
with h5py.File('preprocessed_data.h5', 'w') as f:
    f.create_dataset('walking', data=balanced_walking_data)
    f.create_dataset('jumping', data=balanced_jumping_data)
    f.create_dataset('normalized', data=normalized_data)